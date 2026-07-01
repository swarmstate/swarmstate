//! Stable, language-agnostic serialization for state values.
//!
//! Values are encoded to **msgpack** so that state written by one framework
//! (or language) can be read by another. We walk Python objects directly into
//! an `rmpv::Value` and back, supporting the JSON-like core types plus `bytes`
//! and `tuple`.
//!
//! Supported types: `None`, `bool`, `int` (64-bit), `float`, `str`, `bytes`,
//! `list`, `tuple` (decoded back as `list`), and `dict`.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use rmpv::Value;

/// Convert a Python object into an `rmpv::Value`.
pub fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Nil);
    }
    // `bool` is a subclass of `int`, so it must be checked first.
    if let Ok(b) = obj.downcast::<PyBool>() {
        return Ok(Value::Boolean(b.is_true()));
    }
    if let Ok(i) = obj.downcast::<PyInt>() {
        if let Ok(v) = i.extract::<i64>() {
            return Ok(Value::Integer(v.into()));
        }
        if let Ok(v) = i.extract::<u64>() {
            return Ok(Value::Integer(v.into()));
        }
        return Err(PyValueError::new_err(
            "integer out of 64-bit range is not supported by the swarmstate codec",
        ));
    }
    if let Ok(f) = obj.downcast::<PyFloat>() {
        return Ok(Value::F64(f.value()));
    }
    if let Ok(s) = obj.downcast::<PyString>() {
        return Ok(Value::String(s.extract::<String>()?.into()));
    }
    if let Ok(b) = obj.downcast::<PyBytes>() {
        return Ok(Value::Binary(b.as_bytes().to_vec()));
    }
    if let Ok(d) = obj.downcast::<PyDict>() {
        let mut pairs = Vec::with_capacity(d.len());
        for (k, v) in d.iter() {
            pairs.push((py_to_value(&k)?, py_to_value(&v)?));
        }
        return Ok(Value::Map(pairs));
    }
    if let Ok(list) = obj.downcast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::Array(items));
    }
    if let Ok(tuple) = obj.downcast::<PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::Array(items));
    }
    Err(PyTypeError::new_err(
        "unsupported value type for the swarmstate msgpack codec \
         (supported: None, bool, int, float, str, bytes, list, tuple, dict)",
    ))
}

/// Convert an `rmpv::Value` back into a Python object.
pub fn value_to_py<'py>(py: Python<'py>, val: &Value) -> PyResult<Bound<'py, PyAny>> {
    let obj = match val {
        Value::Nil => py.None().into_bound(py),
        Value::Boolean(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Integer(i) => {
            if let Some(v) = i.as_i64() {
                v.into_pyobject(py)?.into_any()
            } else if let Some(v) = i.as_u64() {
                v.into_pyobject(py)?.into_any()
            } else {
                return Err(PyValueError::new_err("invalid integer in msgpack data"));
            }
        }
        Value::F32(f) => (f64::from(*f)).into_pyobject(py)?.into_any(),
        Value::F64(f) => f.into_pyobject(py)?.into_any(),
        Value::String(s) => match s.as_str() {
            Some(st) => st.into_pyobject(py)?.into_any(),
            // Non-UTF8 msgpack string: surface the raw bytes rather than fail.
            None => PyBytes::new(py, s.as_bytes()).into_any(),
        },
        Value::Binary(b) => PyBytes::new(py, b).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for it in items {
                list.append(value_to_py(py, it)?)?;
            }
            list.into_any()
        }
        Value::Map(pairs) => {
            let dict = PyDict::new(py);
            for (k, v) in pairs {
                dict.set_item(value_to_py(py, k)?, value_to_py(py, v)?)?;
            }
            dict.into_any()
        }
        Value::Ext(_, _) => {
            return Err(PyValueError::new_err(
                "msgpack extension types are not supported by the swarmstate codec",
            ));
        }
    };
    Ok(obj)
}

/// Encode a Python object to msgpack bytes.
pub fn encode(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let val = py_to_value(obj)?;
    let mut buf = Vec::new();
    rmpv::encode::write_value(&mut buf, &val)
        .map_err(|e| PyValueError::new_err(format!("msgpack encode error: {e}")))?;
    Ok(buf)
}

/// Decode msgpack bytes back into a Python object.
pub fn decode<'py>(py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyAny>> {
    let mut cursor = std::io::Cursor::new(bytes);
    let val = rmpv::decode::read_value(&mut cursor)
        .map_err(|e| PyValueError::new_err(format!("msgpack decode error: {e}")))?;
    value_to_py(py, &val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_core_types() {
        Python::with_gil(|py| {
            let src = PyDict::new(py);
            src.set_item("step", 3i64).unwrap();
            src.set_item("ratio", 1.5f64).unwrap();
            src.set_item("name", "onboarding").unwrap();
            src.set_item("done", false).unwrap();
            src.set_item("tags", vec!["a", "b"]).unwrap();
            src.set_item("nothing", py.None()).unwrap();

            let bytes = encode(src.as_any()).unwrap();
            let back = decode(py, &bytes).unwrap();
            let back = back.downcast::<PyDict>().unwrap();

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
    fn bytes_roundtrip_preserved() {
        Python::with_gil(|py| {
            let data = PyBytes::new(py, &[0u8, 1, 2, 255]);
            let bytes = encode(data.as_any()).unwrap();
            let back = decode(py, &bytes).unwrap();
            assert_eq!(
                back.downcast::<PyBytes>().unwrap().as_bytes(),
                &[0u8, 1, 2, 255]
            );
        });
    }
}
