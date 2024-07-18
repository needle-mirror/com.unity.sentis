// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace SentisFlatBuffer
{

using global::System;
using global::System.Collections.Generic;
using global::Unity.Sentis.Google.FlatBuffers;

struct EDim : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_23_5_26(); }
  public static EDim GetRootAsEDim(ByteBuffer _bb) { return GetRootAsEDim(_bb, new EDim()); }
  public static EDim GetRootAsEDim(ByteBuffer _bb, EDim obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public EDim __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public SentisFlatBuffer.SymbolicDim ValType { get { int o = __p.__offset(4); return o != 0 ? (SentisFlatBuffer.SymbolicDim)__p.bb.Get(o + __p.bb_pos) : SentisFlatBuffer.SymbolicDim.NONE; } }
  public TTable? Val<TTable>() where TTable : struct, IFlatbufferObject { int o = __p.__offset(6); return o != 0 ? (TTable?)__p.__union<TTable>(o + __p.bb_pos) : null; }
  public SentisFlatBuffer.Int ValAsInt() { return Val<SentisFlatBuffer.Int>().Value; }
  public SentisFlatBuffer.Byte ValAsByte() { return Val<SentisFlatBuffer.Byte>().Value; }

  public static Offset<SentisFlatBuffer.EDim> CreateEDim(FlatBufferBuilder builder,
      SentisFlatBuffer.SymbolicDim val_type = SentisFlatBuffer.SymbolicDim.NONE,
      int valOffset = 0) {
    builder.StartTable(2);
    EDim.AddVal(builder, valOffset);
    EDim.AddValType(builder, val_type);
    return EDim.EndEDim(builder);
  }

  public static void StartEDim(FlatBufferBuilder builder) { builder.StartTable(2); }
  public static void AddValType(FlatBufferBuilder builder, SentisFlatBuffer.SymbolicDim valType) { builder.AddByte(0, (byte)valType, 0); }
  public static void AddVal(FlatBufferBuilder builder, int valOffset) { builder.AddOffset(1, valOffset, 0); }
  public static Offset<SentisFlatBuffer.EDim> EndEDim(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<SentisFlatBuffer.EDim>(o);
  }
}


static class EDimVerify
{
  static public bool Verify(Unity.Sentis.Google.FlatBuffers.Verifier verifier, uint tablePos)
  {
    return verifier.VerifyTableStart(tablePos)
      && verifier.VerifyField(tablePos, 4 /*ValType*/, 1 /*SentisFlatBuffer.SymbolicDim*/, 1, false)
      && verifier.VerifyUnion(tablePos, 4, 6 /*Val*/, SentisFlatBuffer.SymbolicDimVerify.Verify, false)
      && verifier.VerifyTableEnd(tablePos);
  }
}

}
