//! A small, safe expression evaluator for `HandoffGraph` edge conditions.
//!
//! Conditions like `category == 'billing' and priority > 3` are parsed into an
//! AST once (at `add_edge` time) and evaluated against the routing state. This
//! is a **bounded mini-language**, never Python `eval`:
//!
//! - literals: strings (`'x'`/`"x"`), ints, floats, `true`/`false`, `null`
//! - state access: identifiers with dotted paths (`data.user.role`)
//! - comparisons: `== != < <= > >=`, and membership `in`
//! - logic: `and`, `or`, `not`, and parentheses
//!
//! Evaluation is total: type mismatches yield `false` rather than raising, so
//! routing is deterministic and can never crash on unexpected state.

use rmpv::Value;

const MAX_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    And,
    Or,
    Not,
    In,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LParen,
    RParen,
    Dot,
}

/// Parsed condition expression.
#[derive(Debug, Clone)]
pub enum Expr {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Null,
    Var(Vec<String>),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Cmp(CmpOp, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
}

// ------------------------------------------------------------------ lexer

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        match c {
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            '.' => {
                toks.push(Tok::Dot);
                i += 1;
            }
            '\'' | '"' => {
                let quote = c;
                i += 1;
                let mut s = String::new();
                let mut closed = false;
                while i < chars.len() {
                    let ch = chars[i];
                    if ch == '\\' && i + 1 < chars.len() {
                        let next = chars[i + 1];
                        s.push(match next {
                            'n' => '\n',
                            't' => '\t',
                            '\\' => '\\',
                            '\'' => '\'',
                            '"' => '"',
                            other => other,
                        });
                        i += 2;
                        continue;
                    }
                    if ch == quote {
                        closed = true;
                        i += 1;
                        break;
                    }
                    s.push(ch);
                    i += 1;
                }
                if !closed {
                    return Err("unterminated string literal".to_string());
                }
                toks.push(Tok::Str(s));
            }
            '=' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::Eq);
                    i += 2;
                } else {
                    return Err("expected '==' (single '=' is not valid)".to_string());
                }
            }
            '!' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::Ne);
                    i += 2;
                } else {
                    return Err("expected '!=' (use 'not' for negation)".to_string());
                }
            }
            '<' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::Le);
                    i += 2;
                } else {
                    toks.push(Tok::Lt);
                    i += 1;
                }
            }
            '>' => {
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    toks.push(Tok::Ge);
                    i += 2;
                } else {
                    toks.push(Tok::Gt);
                    i += 1;
                }
            }
            // Number (optionally negative, since '-' is not otherwise used).
            '0'..='9' => {
                let (tok, ni) = lex_number(&chars, i)?;
                toks.push(tok);
                i = ni;
            }
            '-' if i + 1 < chars.len() && chars[i + 1].is_ascii_digit() => {
                let (tok, ni) = lex_number(&chars, i)?;
                toks.push(tok);
                i = ni;
            }
            c if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                toks.push(match word.as_str() {
                    "and" => Tok::And,
                    "or" => Tok::Or,
                    "not" => Tok::Not,
                    "in" => Tok::In,
                    "true" | "True" => Tok::Bool(true),
                    "false" | "False" => Tok::Bool(false),
                    "null" | "none" | "None" => Tok::Null,
                    _ => Tok::Ident(word),
                });
            }
            other => return Err(format!("unexpected character '{other}'")),
        }
    }
    Ok(toks)
}

fn lex_number(chars: &[char], start: usize) -> Result<(Tok, usize), String> {
    let mut i = start;
    if chars[i] == '-' {
        i += 1;
    }
    let mut is_float = false;
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_ascii_digit() {
            i += 1;
        } else if ch == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit() {
            is_float = true;
            i += 1;
        } else {
            break;
        }
    }
    let text: String = chars[start..i].iter().collect();
    if is_float {
        text.parse::<f64>()
            .map(|f| (Tok::Float(f), i))
            .map_err(|_| format!("invalid number '{text}'"))
    } else {
        text.parse::<i64>()
            .map(|n| (Tok::Int(n), i))
            .map_err(|_| format!("invalid number '{text}'"))
    }
}

// ------------------------------------------------------------------ parser

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn parse_or(&mut self, depth: usize) -> Result<Expr, String> {
        if depth > MAX_DEPTH {
            return Err("condition nested too deeply".to_string());
        }
        let mut left = self.parse_and(depth + 1)?;
        while matches!(self.peek(), Some(Tok::Or)) {
            self.pos += 1;
            let right = self.parse_and(depth + 1)?;
            left = Expr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self, depth: usize) -> Result<Expr, String> {
        let mut left = self.parse_not(depth + 1)?;
        while matches!(self.peek(), Some(Tok::And)) {
            self.pos += 1;
            let right = self.parse_not(depth + 1)?;
            left = Expr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self, depth: usize) -> Result<Expr, String> {
        if matches!(self.peek(), Some(Tok::Not)) {
            self.pos += 1;
            let inner = self.parse_not(depth + 1)?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.parse_cmp(depth + 1)
    }

    fn parse_cmp(&mut self, depth: usize) -> Result<Expr, String> {
        let left = self.parse_primary(depth + 1)?;
        let op = match self.peek() {
            Some(Tok::Eq) => CmpOp::Eq,
            Some(Tok::Ne) => CmpOp::Ne,
            Some(Tok::Lt) => CmpOp::Lt,
            Some(Tok::Le) => CmpOp::Le,
            Some(Tok::Gt) => CmpOp::Gt,
            Some(Tok::Ge) => CmpOp::Ge,
            Some(Tok::In) => CmpOp::In,
            _ => return Ok(left),
        };
        self.pos += 1;
        let right = self.parse_primary(depth + 1)?;
        Ok(Expr::Cmp(op, Box::new(left), Box::new(right)))
    }

    fn parse_primary(&mut self, depth: usize) -> Result<Expr, String> {
        if depth > MAX_DEPTH {
            return Err("condition nested too deeply".to_string());
        }
        match self.next() {
            Some(Tok::LParen) => {
                let e = self.parse_or(depth + 1)?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err("expected ')'".to_string()),
                }
            }
            Some(Tok::Str(s)) => Ok(Expr::Str(s)),
            Some(Tok::Int(n)) => Ok(Expr::Int(n)),
            Some(Tok::Float(f)) => Ok(Expr::Float(f)),
            Some(Tok::Bool(b)) => Ok(Expr::Bool(b)),
            Some(Tok::Null) => Ok(Expr::Null),
            Some(Tok::Ident(first)) => {
                let mut path = vec![first];
                while matches!(self.peek(), Some(Tok::Dot)) {
                    self.pos += 1;
                    match self.next() {
                        Some(Tok::Ident(seg)) => path.push(seg),
                        _ => return Err("expected identifier after '.'".to_string()),
                    }
                }
                Ok(Expr::Var(path))
            }
            other => Err(format!("unexpected token {other:?}")),
        }
    }
}

/// Parse a condition string into an [`Expr`], validating syntax eagerly.
pub fn parse(src: &str) -> Result<Expr, String> {
    let toks = lex(src)?;
    if toks.is_empty() {
        return Err("empty condition".to_string());
    }
    let mut p = Parser { toks, pos: 0 };
    let expr = p.parse_or(0)?;
    if p.pos != p.toks.len() {
        return Err(format!("unexpected trailing token {:?}", p.toks[p.pos]));
    }
    Ok(expr)
}

// --------------------------------------------------------------- evaluate

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(i) => i
            .as_i64()
            .map(|x| x as f64)
            .or_else(|| i.as_u64().map(|x| x as f64)),
        Value::F64(f) => Some(*f),
        Value::F32(f) => Some(f64::from(*f)),
        _ => None,
    }
}

/// Truthiness of a value (Python-like).
fn truthy(v: &Value) -> bool {
    match v {
        Value::Nil => false,
        Value::Boolean(b) => *b,
        Value::Integer(_) | Value::F32(_) | Value::F64(_) => {
            as_f64(v).map(|f| f != 0.0).unwrap_or(false)
        }
        Value::String(s) => !s.as_str().map(str::is_empty).unwrap_or(true),
        Value::Binary(b) => !b.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Map(m) => !m.is_empty(),
        Value::Ext(_, _) => true,
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    if let (Some(x), Some(y)) = (as_f64(a), as_f64(b)) {
        return x == y;
    }
    a == b
}

fn ordering_true(op: CmpOp, a: &Value, b: &Value) -> bool {
    if let (Some(x), Some(y)) = (as_f64(a), as_f64(b)) {
        return match op {
            CmpOp::Lt => x < y,
            CmpOp::Le => x <= y,
            CmpOp::Gt => x > y,
            CmpOp::Ge => x >= y,
            _ => false,
        };
    }
    if let (Value::String(x), Value::String(y)) = (a, b) {
        if let (Some(x), Some(y)) = (x.as_str(), y.as_str()) {
            return match op {
                CmpOp::Lt => x < y,
                CmpOp::Le => x <= y,
                CmpOp::Gt => x > y,
                CmpOp::Ge => x >= y,
                _ => false,
            };
        }
    }
    false
}

fn is_member(a: &Value, b: &Value) -> bool {
    match b {
        Value::Array(items) => items.iter().any(|it| values_equal(it, a)),
        Value::String(hay) => match (hay.as_str(), a) {
            (Some(h), Value::String(needle)) => {
                needle.as_str().map(|n| h.contains(n)).unwrap_or(false)
            }
            _ => false,
        },
        Value::Map(pairs) => pairs.iter().any(|(k, _)| values_equal(k, a)),
        _ => false,
    }
}

fn lookup<'a>(state: &'a Value, path: &[String]) -> Option<&'a Value> {
    let mut cur = state;
    for seg in path {
        match cur {
            Value::Map(pairs) => {
                let found = pairs.iter().find(|(k, _)| match k {
                    Value::String(s) => s.as_str() == Some(seg.as_str()),
                    _ => false,
                });
                cur = &found?.1;
            }
            _ => return None,
        }
    }
    Some(cur)
}

fn eval(expr: &Expr, state: &Value) -> Value {
    match expr {
        Expr::Str(s) => Value::String(s.clone().into()),
        Expr::Int(n) => Value::Integer((*n).into()),
        Expr::Float(f) => Value::F64(*f),
        Expr::Bool(b) => Value::Boolean(*b),
        Expr::Null => Value::Nil,
        Expr::Var(path) => lookup(state, path).cloned().unwrap_or(Value::Nil),
        Expr::Not(e) => Value::Boolean(!truthy(&eval(e, state))),
        Expr::And(a, b) => {
            if !truthy(&eval(a, state)) {
                Value::Boolean(false)
            } else {
                Value::Boolean(truthy(&eval(b, state)))
            }
        }
        Expr::Or(a, b) => {
            if truthy(&eval(a, state)) {
                Value::Boolean(true)
            } else {
                Value::Boolean(truthy(&eval(b, state)))
            }
        }
        Expr::Cmp(op, a, b) => {
            let av = eval(a, state);
            let bv = eval(b, state);
            let res = match op {
                CmpOp::Eq => values_equal(&av, &bv),
                CmpOp::Ne => !values_equal(&av, &bv),
                CmpOp::In => is_member(&av, &bv),
                other => ordering_true(*other, &av, &bv),
            };
            Value::Boolean(res)
        }
    }
}

/// Evaluate a parsed condition against `state`, returning its truthiness.
pub fn eval_truthy(expr: &Expr, state: &Value) -> bool {
    truthy(&eval(expr, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(pairs: Vec<(&str, Value)>) -> Value {
        Value::Map(
            pairs
                .into_iter()
                .map(|(k, v)| (Value::String(k.into()), v))
                .collect(),
        )
    }

    fn check(cond: &str, st: &Value) -> bool {
        eval_truthy(&parse(cond).unwrap(), st)
    }

    #[test]
    fn string_equality() {
        let s = state(vec![("category", Value::String("billing".into()))]);
        assert!(check("category == 'billing'", &s));
        assert!(!check("category == 'refund'", &s));
        assert!(check("category != 'refund'", &s));
    }

    #[test]
    fn numeric_and_logic() {
        let s = state(vec![
            ("priority", Value::Integer(5.into())),
            ("vip", Value::Boolean(true)),
        ]);
        assert!(check("priority > 3", &s));
        assert!(check("priority >= 5 and vip", &s));
        assert!(!check("priority < 3 or not vip", &s));
        assert!(check("priority == 5", &s)); // int vs int
    }

    #[test]
    fn dotted_paths_and_missing() {
        let inner = state(vec![("role", Value::String("admin".into()))]);
        let s = state(vec![("user", inner)]);
        assert!(check("user.role == 'admin'", &s));
        assert!(!check("user.role == 'guest'", &s));
        // Missing path -> Nil -> comparison false, truthiness false.
        assert!(!check("user.missing == 'x'", &s));
        assert!(!check("nope", &s));
    }

    #[test]
    fn membership_and_parens() {
        let s = state(vec![
            ("tag", Value::String("urgent".into())),
            (
                "tags",
                Value::Array(vec![
                    Value::String("a".into()),
                    Value::String("urgent".into()),
                ]),
            ),
        ]);
        assert!(check("tag in tags", &s));
        assert!(check("'urgent' in tags", &s));
        assert!(!check("'missing' in tags", &s));
        assert!(check("(tag == 'urgent' or tag == 'x') and 'a' in tags", &s));
    }

    #[test]
    fn cross_int_float_equality() {
        let s = state(vec![("n", Value::F64(5.0))]);
        assert!(check("n == 5", &s));
        assert!(check("n >= 5 and n <= 5", &s));
    }

    #[test]
    fn syntax_errors() {
        assert!(parse("category = 'x'").is_err()); // single '='
        assert!(parse("").is_err());
        assert!(parse("(a == 1").is_err()); // unbalanced
        assert!(parse("a == ").is_err());
    }
}
