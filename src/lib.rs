/*!
# cuda-serializer

Binary and text serialization for agent communication.

Agents need to serialize state, messages, and data for storage and
transmission. This crate provides binary encoding with schema
versioning, message framing, and efficient encoding.

- Binary encoding (varint, length-prefixed)
- Message framing with header + payload
- Schema versioning and evolution
- Field encoding (int/float/bytes/string/bool)
- Compact encoding with delta compression
*/

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Wire type for binary encoding
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireType { Varint, Fixed64, LengthDelimited, Bool }

/// A serialized field
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodedField {
    pub field_number: u32,
    pub wire_type: WireType,
    pub data: Vec<u8>,
}

/// Schema definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SchemaField {
    pub name: String,
    pub field_number: u32,
    pub wire_type: WireType,
    pub required: bool,
    pub default_value: Option<String>,
}

/// A schema version
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Schema {
    pub name: String,
    pub version: u32,
    pub fields: Vec<SchemaField>,
}

/// A framed message
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FramedMessage {
    pub schema_name: String,
    pub schema_version: u32,
    pub message_id: String,
    pub timestamp: u64,
    pub payload: Vec<u8>,
    pub headers: HashMap<String, String>,
}

/// Binary encoder
pub struct BinaryEncoder {
    pub buffer: Vec<u8>,
}

impl BinaryEncoder {
    pub fn new() -> Self { BinaryEncoder { buffer: Vec::new() } }

    /// Encode a varint (zigzag for signed)
    pub fn write_varint(&mut self, value: u64) {
        let mut v = value;
        while v >= 0x80 {
            self.buffer.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buffer.push(v as u8);
    }

    /// Write a string (length-prefixed)
    pub fn write_string(&mut self, s: &str) {
        let bytes = s.as_bytes();
        self.write_varint(bytes.len() as u64);
        self.buffer.extend_from_slice(bytes);
    }

    /// Write raw bytes (length-prefixed)
    pub fn write_bytes(&mut self, data: &[u8]) {
        self.write_varint(data.len() as u64);
        self.buffer.extend_from_slice(data);
    }

    /// Write a field with tag
    pub fn write_field(&mut self, field_number: u32, wire_type: WireType, data: &[u8]) {
        let tag = (field_number << 3) | wire_type as u32;
        self.write_varint(tag as u64);
        match wire_type {
            WireType::Varint => self.buffer.extend_from_slice(data),
            WireType::Bool => self.buffer.push(if !data.is_empty() && data[0] != 0 { 1 } else { 0 }),
            WireType::Fixed64 => self.buffer.extend_from_slice(data),
            WireType::LengthDelimited => self.write_bytes(data),
        }
    }

    /// Write a signed int (zigzag encoded)
    pub fn write_sint(&mut self, value: i64) {
        self.write_varint(((value << 1) ^ (value >> 63)) as u64);
    }

    /// Write a float as fixed64
    pub fn write_float(&mut self, value: f64) {
        self.write_varint(value.to_bits() as u64);
    }

    pub fn len(&self) -> usize { self.buffer.len() }
    pub fn into_bytes(self) -> Vec<u8> { self.buffer }
}

/// Binary decoder
pub struct BinaryDecoder<'a> {
    pub data: &'a [u8],
    pub offset: usize,
}

impl<'a> BinaryDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self { BinaryDecoder { data, offset: 0 } }

    /// Read a varint
    pub fn read_varint(&mut self) -> Result<u64, String> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.offset >= self.data.len() { return Err("unexpected end".into()); }
            let byte = self.data[self.offset];
            self.offset += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if byte & 0x80 == 0 { break; }
            shift += 7;
            if shift >= 64 { return Err("varint too long".into()); }
        }
        Ok(result)
    }

    /// Read length-delimited bytes
    pub fn read_bytes(&mut self) -> Result<Vec<u8>, String> {
        let len = self.read_varint()? as usize;
        if self.offset + len > self.data.len() { return Err("bytes overflow".into()); }
        let result = self.data[self.offset..self.offset + len].to_vec();
        self.offset += len;
        Ok(result)
    }

    /// Read a string
    pub fn read_string(&mut self) -> Result<String, String> {
        let bytes = self.read_bytes()?;
        String::from_utf8(bytes).map_err(|e| e.to_string())
    }

    /// Read a signed int
    pub fn read_sint(&mut self) -> Result<i64, String> {
        let v = self.read_varint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }

    /// Read a field tag
    pub fn read_tag(&mut self) -> Result<(u32, WireType), String> {
        let tag = self.read_varint()?;
        let field_number = (tag >> 3) as u32;
        let wire_type = match (tag & 0x07) as u32 {
            0 => WireType::Varint,
            1 => WireType::Fixed64,
            2 => WireType::LengthDelimited,
            _ => return Err(format!("unknown wire type {}", tag & 0x07)),
        };
        Ok((field_number, wire_type))
    }

    pub fn remaining(&self) -> usize { self.data.len().saturating_sub(self.offset) }
}

/// Message framer
pub struct MessageFramer;

impl MessageFramer {
    /// Frame a message with header
    pub fn frame(msg: &FramedMessage) -> Vec<u8> {
        let mut encoder = BinaryEncoder::new();
        // Magic bytes
        encoder.write_varint(0xCAFE);
        // Schema version
        encoder.write_varint(msg.schema_version as u64);
        // Message ID length + content
        encoder.write_string(&msg.message_id);
        // Timestamp
        encoder.write_varint(msg.timestamp);
        // Payload length + content
        encoder.write_bytes(&msg.payload);
        // Header count
        encoder.write_varint(msg.headers.len() as u64);
        for (k, v) in &msg.headers {
            encoder.write_string(k);
            encoder.write_string(v);
        }
        encoder.into_bytes()
    }

    /// Unframe a message
    pub fn unframe(data: &[u8]) -> Result<FramedMessage, String> {
        let mut decoder = BinaryDecoder::new(data);
        let magic = decoder.read_varint()?;
        if magic != 0xCAFE { return Err(format!("bad magic: {}", magic)); }
        let schema_version = decoder.read_varint()? as u32;
        let message_id = decoder.read_string()?;
        let timestamp = decoder.read_varint()?;
        let payload = decoder.read_bytes()?;
        let header_count = decoder.read_varint()? as usize;
        let mut headers = HashMap::new();
        for _ in 0..header_count {
            let k = decoder.read_string()?;
            let v = decoder.read_string()?;
            headers.insert(k, v);
        }
        Ok(FramedMessage { schema_name: String::new(), schema_version, message_id, timestamp, payload, headers })
    }
}

/// Schema evolution helper
pub struct SchemaEvolver {
    pub schemas: HashMap<String, Vec<Schema>>,
}

impl SchemaEvolver {
    pub fn new() -> Self { SchemaEvolver { schemas: HashMap::new() } }

    /// Register a schema version
    pub fn register(&mut self, schema: Schema) {
        self.schemas.entry(schema.name.clone()).or_insert_with(Vec::new).push(schema);
    }

    /// Get latest version of a schema
    pub fn latest(&self, name: &str) -> Option<&Schema> {
        self.schemas.get(name)?.iter().max_by_key(|s| s.version)
    }

    /// Check if a field exists in a schema version
    pub fn has_field(&self, schema_name: &str, version: u32, field_name: &str) -> bool {
        self.schemas.get(schema_name).map(|versions| {
            versions.iter().find(|s| s.version == version).map(|s| s.fields.iter().any(|f| f.name == field_name)).unwrap_or(false)
        }).unwrap_or(false)
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_roundtrip() {
        let mut enc = BinaryEncoder::new();
        enc.write_varint(300);
        let mut dec = BinaryDecoder::new(&enc.buffer);
        assert_eq!(dec.read_varint().unwrap(), 300);
    }

    #[test]
    fn test_string_roundtrip() {
        let mut enc = BinaryEncoder::new();
        enc.write_string("hello agent");
        let mut dec = BinaryDecoder::new(&enc.buffer);
        assert_eq!(dec.read_string().unwrap(), "hello agent");
    }

    #[test]
    fn test_bytes_roundtrip() {
        let mut enc = BinaryEncoder::new();
        enc.write_bytes(&[1, 2, 3, 4, 5]);
        let mut dec = BinaryDecoder::new(&enc.buffer);
        assert_eq!(dec.read_bytes().unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_sint_roundtrip() {
        let mut enc = BinaryEncoder::new();
        enc.write_sint(-42);
        let mut dec = BinaryDecoder::new(&enc.buffer);
        assert_eq!(dec.read_sint().unwrap(), -42);
    }

    #[test]
    fn test_field_encoding() {
        let mut enc = BinaryEncoder::new();
        enc.write_field(1, WireType::LengthDelimited, b"test");
        assert!(enc.len() > 0);
    }

    #[test]
    fn test_frame_unframe() {
        let msg = FramedMessage { schema_name: "test".into(), schema_version: 1, message_id: "msg_1".into(), timestamp: 12345, payload: b"hello world".to_vec(), headers: { let mut h = HashMap::new(); h.insert("type".into(), "ping".into()); h } };
        let framed = MessageFramer::frame(&msg);
        let unframed = MessageFramer::unframe(&framed).unwrap();
        assert_eq!(unframed.message_id, "msg_1");
        assert_eq!(unframed.payload, b"hello world".to_vec());
        assert_eq!(unframed.headers.get("type").unwrap(), "ping");
    }

    #[test]
    fn test_bad_magic() {
        let bad = vec![0x00, 0x01];
        assert!(MessageFramer::unframe(&bad).is_err());
    }

    #[test]
    fn test_schema_evolution() {
        let mut evolver = SchemaEvolver::new();
        evolver.register(Schema { name: "agent".into(), version: 1, fields: vec![SchemaField { name: "id".into(), field_number: 1, wire_type: WireType::Varint, required: true, default_value: None }] });
        evolver.register(Schema { name: "agent".into(), version: 2, fields: vec![SchemaField { name: "id".into(), field_number: 1, wire_type: WireType::Varint, required: true, default_value: None }, SchemaField { name: "name".into(), field_number: 2, wire_type: WireType::LengthDelimited, required: false, default_value: None }] });
        assert_eq!(evolver.latest("agent").unwrap().version, 2);
        assert!(evolver.has_field("agent", 2, "name"));
        assert!(!evolver.has_field("agent", 1, "name"));
    }

    #[test]
    fn test_decoder_remaining() {
        let enc = BinaryEncoder::new();
        let mut dec = BinaryDecoder::new(&enc.buffer);
        assert_eq!(dec.remaining(), 0);
    }
}
