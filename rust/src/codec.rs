//! Stable, language-agnostic serialization for state values.
//!
//! Values are encoded to **msgpack** so that state written by one framework
//! (or language) can be read by another. Supported types: `None`, `bool`,
//! `int` (64-bit), `float`, `str`, `bytes`, `list`, `tuple` (decoded back as
//! `list`), and `dict`.
//!
//! Both directions stream: [`encode`] writes msgpack bytes straight from the
//! Python objects and [`decode`] builds Python objects straight from the byte
//! slice, with no intermediate `rmpv::Value` tree. That matters because the tree
//! copied every payload twice on the way out and twice on the way back — pure
//! overhead on the store's hot path. The byte format is unchanged: the encoder
//! picks the same representations `rmpv` did (see the interop tests, which check
//! swarmstate's bytes against an independent msgpack implementation).
//!
//! [`py_to_value`] still produces an `rmpv::Value`, for the condition evaluator
//! in [`crate::condition`] — that is a lookup structure, not a wire format.
//!
//! Both walkers recurse, so both are bounded by [`MAX_DEPTH`]: without it, a
//! deeply nested (or self-referential) value would exhaust the native stack and
//! abort the whole process instead of raising.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use rmpv::Value;

/// Maximum nesting depth accepted by the codec, in either direction.
///
/// 128 is the same budget `serde_json` allows, and is chosen against the *stack*
/// rather than the format: these walkers recurse once per level, and a Python
/// worker thread (e.g. the `asyncio.to_thread` calls in the LangGraph adapter)
/// can have a stack of only a few hundred KiB. It is also well below
/// `rmpv::decode::MAX_DEPTH` (1024), so anything we encode decodes back.
/// Real agent state nests a handful of levels; 128 only ever trips on runaway
/// or self-referential structures.
pub const MAX_DEPTH: usize = 128;

fn too_deep() -> PyErr {
    PyValueError::new_err(format!(
        "value nests deeper than {MAX_DEPTH} levels (or is self-referential); \
         the swarmstate msgpack codec rejects it to protect the native stack"
    ))
}

fn unsupported_type() -> PyErr {
    PyTypeError::new_err(
        "unsupported value type for the swarmstate msgpack codec \
         (supported: None, bool, int, float, str, bytes, list, tuple, dict)",
    )
}

fn int_out_of_range() -> PyErr {
    PyValueError::new_err("integer out of 64-bit range is not supported by the swarmstate codec")
}

fn malformed(what: &str) -> PyErr {
    PyValueError::new_err(format!("msgpack decode error: {what}"))
}

// ----------------------------------------------------------------- encoding

/// Encode a Python object to msgpack bytes.
pub fn encode(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let mut buf = Vec::with_capacity(64);
    write_obj(&mut buf, obj, 0)?;
    Ok(buf)
}

/// Write `obj` to `buf` as msgpack, recursing through containers.
fn write_obj(buf: &mut Vec<u8>, obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<()> {
    if depth > MAX_DEPTH {
        return Err(too_deep());
    }
    // Ordered by how often each type actually shows up in agent state — strings
    // and dicts dominate a message history — because every miss is a type check.
    // The one ordering that is not free to choose: `bool` is a subclass of `int`,
    // so it has to be tested before it.
    if let Ok(s) = obj.cast::<PyString>() {
        // to_cow borrows the interpreter's buffer where it can (to_str is not
        // available under the abi3 limited API we build against).
        let text = s.to_cow()?;
        write_str_header(buf, text.len())?;
        buf.extend_from_slice(text.as_bytes());
        return Ok(());
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        write_map_header(buf, d.len())?;
        for (k, v) in d.iter() {
            write_obj(buf, &k, depth + 1)?;
            write_obj(buf, &v, depth + 1)?;
        }
        return Ok(());
    }
    if let Ok(list) = obj.cast::<PyList>() {
        write_array_header(buf, list.len())?;
        for item in list.iter() {
            write_obj(buf, &item, depth + 1)?;
        }
        return Ok(());
    }
    if obj.is_none() {
        buf.push(0xc0);
        return Ok(());
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        buf.push(if b.is_true() { 0xc3 } else { 0xc2 });
        return Ok(());
    }
    if let Ok(i) = obj.cast::<PyInt>() {
        return match i.extract::<i64>() {
            Ok(v) => write_i64(buf, v),
            // Only values above i64::MAX get here; anything else is out of range.
            Err(_) => match i.extract::<u64>() {
                Ok(v) => write_u64(buf, v),
                Err(_) => Err(int_out_of_range()),
            },
        };
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        buf.push(0xcb);
        buf.extend_from_slice(&f.value().to_bits().to_be_bytes());
        return Ok(());
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        let data = b.as_bytes();
        write_bin_header(buf, data.len())?;
        buf.extend_from_slice(data);
        return Ok(());
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        write_array_header(buf, tuple.len())?;
        for item in tuple.iter() {
            write_obj(buf, &item, depth + 1)?;
        }
        return Ok(());
    }
    Err(unsupported_type())
}

/// Write a signed integer in the shortest msgpack form (matching `rmp`'s choice).
fn write_i64(buf: &mut Vec<u8>, v: i64) -> PyResult<()> {
    match v {
        0..=127 => buf.push(v as u8),
        -32..=-1 => buf.push(v as i8 as u8),
        128..=255 => buf.extend_from_slice(&[0xcc, v as u8]),
        256..=65535 => {
            buf.push(0xcd);
            buf.extend_from_slice(&(v as u16).to_be_bytes());
        }
        65536..=4294967295 => {
            buf.push(0xce);
            buf.extend_from_slice(&(v as u32).to_be_bytes());
        }
        4294967296.. => {
            buf.push(0xcf);
            buf.extend_from_slice(&(v as u64).to_be_bytes());
        }
        -128..=-33 => buf.extend_from_slice(&[0xd0, v as i8 as u8]),
        -32768..=-129 => {
            buf.push(0xd1);
            buf.extend_from_slice(&(v as i16).to_be_bytes());
        }
        -2147483648..=-32769 => {
            buf.push(0xd2);
            buf.extend_from_slice(&(v as i32).to_be_bytes());
        }
        _ => {
            buf.push(0xd3);
            buf.extend_from_slice(&v.to_be_bytes());
        }
    }
    Ok(())
}

/// Write an unsigned integer that did not fit in an `i64`.
fn write_u64(buf: &mut Vec<u8>, v: u64) -> PyResult<()> {
    buf.push(0xcf);
    buf.extend_from_slice(&v.to_be_bytes());
    Ok(())
}

fn write_str_header(buf: &mut Vec<u8>, len: usize) -> PyResult<()> {
    match len {
        0..=31 => buf.push(0xa0 | len as u8),
        32..=255 => buf.extend_from_slice(&[0xd9, len as u8]),
        256..=65535 => {
            buf.push(0xda);
            buf.extend_from_slice(&(len as u16).to_be_bytes());
        }
        _ => {
            buf.push(0xdb);
            buf.extend_from_slice(&u32_len(len, "string")?.to_be_bytes());
        }
    }
    Ok(())
}

fn write_bin_header(buf: &mut Vec<u8>, len: usize) -> PyResult<()> {
    match len {
        0..=255 => buf.extend_from_slice(&[0xc4, len as u8]),
        256..=65535 => {
            buf.push(0xc5);
            buf.extend_from_slice(&(len as u16).to_be_bytes());
        }
        _ => {
            buf.push(0xc6);
            buf.extend_from_slice(&u32_len(len, "bytes")?.to_be_bytes());
        }
    }
    Ok(())
}

fn write_array_header(buf: &mut Vec<u8>, len: usize) -> PyResult<()> {
    match len {
        0..=15 => buf.push(0x90 | len as u8),
        16..=65535 => {
            buf.push(0xdc);
            buf.extend_from_slice(&(len as u16).to_be_bytes());
        }
        _ => {
            buf.push(0xdd);
            buf.extend_from_slice(&u32_len(len, "list")?.to_be_bytes());
        }
    }
    Ok(())
}

fn write_map_header(buf: &mut Vec<u8>, len: usize) -> PyResult<()> {
    match len {
        0..=15 => buf.push(0x80 | len as u8),
        16..=65535 => {
            buf.push(0xde);
            buf.extend_from_slice(&(len as u16).to_be_bytes());
        }
        _ => {
            buf.push(0xdf);
            buf.extend_from_slice(&u32_len(len, "dict")?.to_be_bytes());
        }
    }
    Ok(())
}

/// msgpack lengths are 32-bit; refuse anything larger rather than truncating.
fn u32_len(len: usize, what: &str) -> PyResult<u32> {
    u32::try_from(len).map_err(|_| {
        PyValueError::new_err(format!(
            "{what} of {len} items exceeds the 32-bit length msgpack allows"
        ))
    })
}

// ----------------------------------------------------------------- decoding

/// Decode msgpack bytes back into a Python object.
///
/// Trailing bytes after the first value are ignored, as before.
pub fn decode<'py>(py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let mut rd = Reader {
        data: bytes,
        pos: 0,
    };
    read_obj(py, &mut rd, 0)
}

/// A cursor over the input, handing out borrowed slices (no copies).
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn byte(&mut self) -> PyResult<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| malformed("unexpected end of input"))?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> PyResult<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| malformed("length overflow"))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| malformed("unexpected end of input"))?;
        self.pos = end;
        Ok(slice)
    }

    fn len_of(&mut self, width: usize) -> PyResult<usize> {
        let raw = self.take(width)?;
        Ok(raw.iter().fold(0usize, |acc, b| (acc << 8) | *b as usize))
    }
}

fn read_obj<'py>(
    py: Python<'py>,
    rd: &mut Reader<'_>,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if depth > MAX_DEPTH {
        return Err(too_deep());
    }
    let marker = rd.byte()?;
    let obj = match marker {
        0x00..=0x7f => (marker as i64).into_pyobject(py)?.into_any(),
        0xe0..=0xff => ((marker as i8) as i64).into_pyobject(py)?.into_any(),
        0x80..=0x8f => read_map(py, rd, (marker & 0x0f) as usize, depth)?,
        0x90..=0x9f => read_array(py, rd, (marker & 0x0f) as usize, depth)?,
        0xa0..=0xbf => read_str(py, rd, (marker & 0x1f) as usize)?,
        0xc0 => py.None().into_bound(py),
        0xc2 => PyBool::new(py, false).to_owned().into_any(),
        0xc3 => PyBool::new(py, true).to_owned().into_any(),
        0xc4 => {
            let n = rd.len_of(1)?;
            read_bin(py, rd, n)?
        }
        0xc5 => {
            let n = rd.len_of(2)?;
            read_bin(py, rd, n)?
        }
        0xc6 => {
            let n = rd.len_of(4)?;
            read_bin(py, rd, n)?
        }
        0xca => {
            let raw = rd.take(4)?;
            let bits = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
            (f32::from_bits(bits) as f64).into_pyobject(py)?.into_any()
        }
        0xcb => {
            let raw = rd.take(8)?;
            let mut bits = [0u8; 8];
            bits.copy_from_slice(raw);
            f64::from_bits(u64::from_be_bytes(bits))
                .into_pyobject(py)?
                .into_any()
        }
        0xcc => (rd.byte()? as i64).into_pyobject(py)?.into_any(),
        0xcd => (rd.len_of(2)? as i64).into_pyobject(py)?.into_any(),
        0xce => (rd.len_of(4)? as i64).into_pyobject(py)?.into_any(),
        0xcf => {
            let raw = rd.take(8)?;
            let mut be = [0u8; 8];
            be.copy_from_slice(raw);
            // Values above i64::MAX stay unsigned, as Python ints are unbounded.
            u64::from_be_bytes(be).into_pyobject(py)?.into_any()
        }
        0xd0 => ((rd.byte()? as i8) as i64).into_pyobject(py)?.into_any(),
        0xd1 => {
            let raw = rd.take(2)?;
            (i16::from_be_bytes([raw[0], raw[1]]) as i64)
                .into_pyobject(py)?
                .into_any()
        }
        0xd2 => {
            let raw = rd.take(4)?;
            (i32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64)
                .into_pyobject(py)?
                .into_any()
        }
        0xd3 => {
            let raw = rd.take(8)?;
            let mut be = [0u8; 8];
            be.copy_from_slice(raw);
            i64::from_be_bytes(be).into_pyobject(py)?.into_any()
        }
        0xd9 => {
            let n = rd.len_of(1)?;
            read_str(py, rd, n)?
        }
        0xda => {
            let n = rd.len_of(2)?;
            read_str(py, rd, n)?
        }
        0xdb => {
            let n = rd.len_of(4)?;
            read_str(py, rd, n)?
        }
        0xdc => {
            let n = rd.len_of(2)?;
            read_array(py, rd, n, depth)?
        }
        0xdd => {
            let n = rd.len_of(4)?;
            read_array(py, rd, n, depth)?
        }
        0xde => {
            let n = rd.len_of(2)?;
            read_map(py, rd, n, depth)?
        }
        0xdf => {
            let n = rd.len_of(4)?;
            read_map(py, rd, n, depth)?
        }
        0xc7..=0xc9 | 0xd4..=0xd8 => {
            return Err(PyValueError::new_err(
                "msgpack extension types are not supported by the swarmstate codec",
            ));
        }
        0xc1 => return Err(malformed("reserved marker 0xc1")),
    };
    Ok(obj)
}

fn read_str<'py>(py: Python<'py>, rd: &mut Reader<'_>, len: usize) -> PyResult<Bound<'py, PyAny>> {
    let raw = rd.take(len)?;
    match std::str::from_utf8(raw) {
        Ok(text) => Ok(PyString::new(py, text).into_any()),
        // Non-UTF8 msgpack string: surface the raw bytes rather than fail.
        Err(_) => Ok(PyBytes::new(py, raw).into_any()),
    }
}

fn read_bin<'py>(py: Python<'py>, rd: &mut Reader<'_>, len: usize) -> PyResult<Bound<'py, PyAny>> {
    Ok(PyBytes::new(py, rd.take(len)?).into_any())
}

fn read_array<'py>(
    py: Python<'py>,
    rd: &mut Reader<'_>,
    len: usize,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let mut items = Vec::with_capacity(len.min(1024));
    for _ in 0..len {
        items.push(read_obj(py, rd, depth + 1)?);
    }
    Ok(PyList::new(py, items)?.into_any())
}

fn read_map<'py>(
    py: Python<'py>,
    rd: &mut Reader<'_>,
    len: usize,
    depth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    for _ in 0..len {
        let key = read_obj(py, rd, depth + 1)?;
        let value = read_obj(py, rd, depth + 1)?;
        dict.set_item(key, value)?;
    }
    Ok(dict.into_any())
}

// ------------------------------------------------- Python -> rmpv::Value
// Used by the condition evaluator, which indexes into a Value tree rather than
// reading a byte stream. Not part of the store's hot path.

/// Convert a Python object into an `rmpv::Value`.
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    py_to_value_at(obj, 0)
}

fn py_to_value_at(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<Value> {
    if depth > MAX_DEPTH {
        return Err(too_deep());
    }
    if obj.is_none() {
        return Ok(Value::Nil);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Boolean(b.is_true()));
    }
    if let Ok(i) = obj.cast::<PyInt>() {
        if let Ok(v) = i.extract::<i64>() {
            return Ok(Value::Integer(v.into()));
        }
        if let Ok(v) = i.extract::<u64>() {
            return Ok(Value::Integer(v.into()));
        }
        return Err(int_out_of_range());
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return Ok(Value::F64(f.value()));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.extract::<String>()?.into()));
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(Value::Binary(b.as_bytes().to_vec()));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut pairs = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            pairs.push((
                py_to_value_at(&k, depth + 1)?,
                py_to_value_at(&v, depth + 1)?,
            ));
        }
        return Ok(Value::Map(pairs));
    }
    if let Ok(list) = obj.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_value_at(&item, depth + 1)?);
        }
        return Ok(Value::Array(items));
    }
    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_to_value_at(&item, depth + 1)?);
        }
        return Ok(Value::Array(items));
    }
    Err(unsupported_type())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_core_types() {
        Python::attach(|py| {
            let src = PyDict::new(py);
            src.set_item("step", 3i64).unwrap();
            src.set_item("ratio", 1.5f64).unwrap();
            src.set_item("name", "onboarding").unwrap();
            src.set_item("done", false).unwrap();
            src.set_item("tags", vec!["a", "b"]).unwrap();
            src.set_item("nothing", py.None()).unwrap();

            let bytes = encode(src.as_any()).unwrap();
            let back = decode(py, &bytes).unwrap();
            let back = back.cast::<PyDict>().unwrap();

            assert_eq!(
                back.get_item("step")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                3
            );
            assert_eq!(
                back.get_item("name")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "onboarding"
            );
            assert!(!back
                .get_item("done")
                .unwrap()
                .unwrap()
                .extract::<bool>()
                .unwrap());
            assert!(back.get_item("nothing").unwrap().unwrap().is_none());
        });
    }

    #[test]
    fn nesting_within_limit_roundtrips() {
        Python::attach(|py| {
            // A chain MAX_DEPTH levels deep must still encode *and* decode.
            let inner = PyDict::new(py);
            inner.set_item("leaf", 1i64).unwrap();
            let mut cur = inner;
            for _ in 0..(MAX_DEPTH - 1) {
                let outer = PyDict::new(py);
                outer.set_item("a", &cur).unwrap();
                cur = outer;
            }
            let bytes = encode(cur.as_any()).unwrap();
            assert!(decode(py, &bytes).is_ok());
        });
    }

    #[test]
    fn too_deep_errors_instead_of_overflowing_the_stack() {
        Python::attach(|py| {
            let inner = PyDict::new(py);
            let mut cur = inner;
            for _ in 0..(MAX_DEPTH + 10) {
                let outer = PyDict::new(py);
                outer.set_item("a", &cur).unwrap();
                cur = outer;
            }
            let err = encode(cur.as_any()).unwrap_err();
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn self_referential_value_errors() {
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("self", &d).unwrap();
            assert!(encode(d.as_any()).is_err());
        });
    }

    /// Deterministic xorshift64*, so a failure here is reproducible from the seed.
    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[test]
    fn decoding_arbitrary_bytes_never_panics() {
        // The decoder reads bytes that may have been written by anything at all —
        // another process, another language, a corrupted Redis value. It has to
        // fail with an error, never take the interpreter down with it.
        Python::attach(|py| {
            let mut state = 0x5EED_1234_ABCD_9876;
            for _ in 0..20_000 {
                let len = (next_random(&mut state) % 96) as usize;
                let bytes: Vec<u8> = (0..len)
                    .map(|_| (next_random(&mut state) & 0xff) as u8)
                    .collect();
                if let Ok(obj) = decode(py, &bytes) {
                    // Whatever we accepted, we can write back out and read again.
                    let again = encode(&obj).and_then(|raw| decode(py, &raw));
                    assert!(again.is_ok(), "re-encode failed for {bytes:02x?}");
                }
            }
        });
    }

    #[test]
    fn truncated_and_flipped_encodings_are_rejected_cleanly() {
        Python::attach(|py| {
            let deep = PyDict::new(py);
            deep.set_item("msgs", vec!["a", "b", "c"]).unwrap();
            deep.set_item("blob", PyBytes::new(py, &[7u8; 300]))
                .unwrap();
            deep.set_item("n", -70000i64).unwrap();
            deep.set_item("f", 2.5f64).unwrap();
            let valid = encode(deep.as_any()).unwrap();

            // Every prefix of a valid encoding: incomplete, so it must error, not hang
            // or crash. (The empty input included.)
            for cut in 0..valid.len() {
                let _ = decode(py, &valid[..cut]);
            }
            // Every single-byte corruption: any marker can appear anywhere.
            for pos in 0..valid.len() {
                for flip in [0x01u8, 0x80, 0xff] {
                    let mut mutated = valid.clone();
                    mutated[pos] ^= flip;
                    let _ = decode(py, &mutated);
                }
            }
            // A huge declared length must not preallocate or read out of bounds.
            for header in [
                vec![0xdd, 0xff, 0xff, 0xff, 0xff], // array of ~4 billion
                vec![0xdf, 0xff, 0xff, 0xff, 0xff], // map of ~4 billion
                vec![0xc6, 0xff, 0xff, 0xff, 0xff], // 4 GB binary
                vec![0xdb, 0xff, 0xff, 0xff, 0xff], // 4 GB string
            ] {
                assert!(decode(py, &header).is_err());
            }
        });
    }

    #[test]
    fn bytes_roundtrip_preserved() {
        Python::attach(|py| {
            let data = PyBytes::new(py, &[0u8, 1, 2, 255]);
            let bytes = encode(data.as_any()).unwrap();
            let back = decode(py, &bytes).unwrap();
            assert_eq!(
                back.cast::<PyBytes>().unwrap().as_bytes(),
                &[0u8, 1, 2, 255]
            );
        });
    }
}
