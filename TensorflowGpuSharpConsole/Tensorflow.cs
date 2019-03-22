using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using CharPtr = System.IntPtr;
using LongPtr = System.IntPtr;
using TF_Buffer = System.IntPtr;
using TF_Graph = System.IntPtr;
using TF_ImportGraphDefOptions = System.IntPtr;
using TF_Operation = System.IntPtr;
using TF_Output = System.IntPtr;
using TF_Session = System.IntPtr;
using TF_SessionOptions = System.IntPtr;
using TF_Status = System.IntPtr;
using TF_Tensor = System.IntPtr;
using VoidPtr = System.IntPtr;

namespace RiceCakeSoftware.TensorflowGpuSharpConsole
{
    public class Tensorflow : IDisposable
    {
        private enum TF_DataType
        {
            TF_FLOAT = 1,
            TF_DOUBLE = 2,
            TF_INT32 = 3,  // Int32 tensors are always in 'host' memory.
            TF_UINT8 = 4,
            TF_INT16 = 5,
            TF_INT8 = 6,
            TF_STRING = 7,
            TF_COMPLEX64 = 8,  // Single-precision complex
            TF_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
            TF_INT64 = 9,
            TF_BOOL = 10,
            TF_QINT8 = 11,     // Quantized int8
            TF_QUINT8 = 12,    // Quantized uint8
            TF_QINT32 = 13,    // Quantized int32
            TF_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
            TF_QINT16 = 15,    // Quantized int16
            TF_QUINT16 = 16,   // Quantized uint16
            TF_UINT16 = 17,
            TF_COMPLEX128 = 18,  // Double-precision complex
            TF_HALF = 19,
            TF_RESOURCE = 20,
            TF_VARIANT = 21,
            TF_UINT32 = 22,
            TF_UINT64 = 23,
        }

        private enum TF_Code
        {
            TF_OK = 0,
            TF_CANCELLED = 1,
            TF_UNKNOWN = 2,
            TF_INVALID_ARGUMENT = 3,
            TF_DEADLINE_EXCEEDED = 4,
            TF_NOT_FOUND = 5,
            TF_ALREADY_EXISTS = 6,
            TF_PERMISSION_DENIED = 7,
            TF_UNAUTHENTICATED = 16,
            TF_RESOURCE_EXHAUSTED = 8,
            TF_FAILED_PRECONDITION = 9,
            TF_ABORTED = 10,
            TF_OUT_OF_RANGE = 11,
            TF_UNIMPLEMENTED = 12,
            TF_INTERNAL = 13,
            TF_UNAVAILABLE = 14,
            TF_DATA_LOSS = 15,
        }

        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void BufferDataDeallocator(VoidPtr data, int length);
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        private delegate void TensorDeallocator(VoidPtr data, int length, VoidPtr arg);

        private TF_Status _status = TF_Status.Zero;
        private TF_Buffer _graphDef = TF_Buffer.Zero;
        private TF_Graph _graph = TF_Graph.Zero;
        private TF_Session _session = TF_Session.Zero;

        public string GetVersion()
        {
            return Marshal.PtrToStringAnsi(TF_Version());
        }

        public void CreateSession(string modelFilePath)
        {
            _status = TF_NewStatus();
            TFBuffer buffer = new TFBuffer();
            byte[] data = File.ReadAllBytes(modelFilePath);
            buffer.Data = Marshal.AllocHGlobal(data.Length);
            Marshal.Copy(data, 0, buffer.Data, data.Length);
            buffer.Length = data.Length;
            buffer.Deallocator = (d, l) => { Marshal.FreeHGlobal(d); };
            _graphDef = TF_NewBuffer();
            Marshal.StructureToPtr<TFBuffer>(buffer, _graphDef, false);

            _graph = TF_NewGraph();
            TF_ImportGraphDefOptions graphDefOptions = TF_NewImportGraphDefOptions();
            TF_GraphImportGraphDef(_graph, _graphDef, graphDefOptions, _status);
            TF_DeleteImportGraphDefOptions(graphDefOptions);
            if (TF_GetCode(_status) != TF_Code.TF_OK)
            {
                throw new Exception(Marshal.PtrToStringAnsi(TF_Message(_status)));
            }

            TF_SessionOptions sessionOptions = TF_NewSessionOptions();
            _session = TF_NewSession(_graph, sessionOptions, _status);
            TF_DeleteSessionOptions(sessionOptions);
            if (TF_GetCode(_status) != TF_Code.TF_OK)
            {
                throw new Exception(Marshal.PtrToStringAnsi(TF_Message(_status)));
            }
        }

        public float[] RunSession(float[] pixels, int width, int height, int bpp, string inputOperationName, string outputOperationName, int classes)
        {
            VoidPtr pixelsPtr = Marshal.AllocHGlobal(sizeof(float) * width * height * bpp);
            Marshal.Copy(pixels, 0, pixelsPtr, width * height * bpp);
            float[] probabries = RunSession(pixelsPtr, width, height, bpp, inputOperationName, outputOperationName, classes);
            Marshal.FreeHGlobal(pixelsPtr);
            return probabries;
        }

        public float[] RunSession(VoidPtr pixels, int width, int height, int bpp, string inputOperationName, string outputOperationName, int classes)
        {
            List<byte> dimsByteList = new List<byte>();
            dimsByteList.AddRange(BitConverter.GetBytes((long)bpp));
            dimsByteList.AddRange(BitConverter.GetBytes((long)width));
            dimsByteList.AddRange(BitConverter.GetBytes((long)height));
            dimsByteList.AddRange(BitConverter.GetBytes(1L));
            LongPtr dimsPtr = Marshal.AllocHGlobal(sizeof(long) * 4);
            Marshal.Copy(dimsByteList.ToArray(), 0, dimsPtr, sizeof(long) * 4);
            TF_Tensor inputTensor = TF_NewTensor(TF_DataType.TF_FLOAT, dimsPtr, 4, pixels, sizeof(float) * width * height * 1, (d, l, a) => { Marshal.FreeHGlobal(d); }, VoidPtr.Zero);
            TF_Tensor outputTensor = Marshal.AllocHGlobal(Marshal.SizeOf<TF_Tensor>());
            CharPtr inputOperationNamePtr = Marshal.StringToHGlobalAnsi(inputOperationName);
            TFOutput input = new TFOutput() { Operation = TF_GraphOperationByName(_graph, inputOperationNamePtr), Index = 0, };
            TF_Output inputPtr = Marshal.AllocHGlobal(Marshal.SizeOf<TFOutput>());
            Marshal.StructureToPtr<TFOutput>(input, inputPtr, false);
            CharPtr outputOperationNamePtr = Marshal.StringToHGlobalAnsi(outputOperationName);
            TFOutput output = new TFOutput() { Operation = TF_GraphOperationByName(_graph, outputOperationNamePtr), Index = 0, };
            TF_Output outputPtr = Marshal.AllocHGlobal(Marshal.SizeOf<TFOutput>());
            Marshal.StructureToPtr<TFOutput>(output, outputPtr, false);
            TF_Operation operation = TF_Operation.Zero;
            TF_SessionRun(_session, VoidPtr.Zero, inputPtr, ref inputTensor, 1, outputPtr, ref outputTensor, 1, ref operation, 0, TF_Buffer.Zero, _status);
            float[] probabries = null;
            if (TF_GetCode(_status) == TF_Code.TF_OK)
            {
                VoidPtr result = TF_TensorData(outputTensor);
                probabries = new float[classes];
                Marshal.Copy(result, probabries, 0, classes);
            }

            Marshal.FreeHGlobal(outputOperationNamePtr);
            Marshal.FreeHGlobal(inputOperationNamePtr);
            Marshal.FreeHGlobal(outputPtr);
            Marshal.FreeHGlobal(inputPtr);
            Marshal.FreeHGlobal(dimsPtr);
            return probabries;
        }

        private void CloseSession()
        {
            if (_session != TF_Session.Zero)
            {
                TF_CloseSession(_session, _status);
                TF_DeleteSession(_session, _status);
                _session = TF_Session.Zero;
            }
            if (_graph != TF_Graph.Zero)
            {
                TF_DeleteGraph(_graph);
                _graph = TF_Graph.Zero;
            }
            if (_graphDef != TF_Buffer.Zero)
            {
                TF_DeleteBuffer(_graphDef);
                _graphDef = TF_Buffer.Zero;
            }
            if (_status != TF_Status.Zero)
            {
                TF_DeleteStatus(_status);
                _status = TF_Status.Zero;
            }
        }

        public void Dispose()
        {
            CloseSession();
        }

        [StructLayout(LayoutKind.Sequential)]
        struct TFBuffer
        {
            public VoidPtr Data;
            public int Length;
            [MarshalAs(UnmanagedType.FunctionPtr)] public BufferDataDeallocator Deallocator;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct TFOutput
        {
            public TF_Operation Operation;
            public int Index;
        }

        [DllImport("tensorflow.dll")]
        private extern static CharPtr TF_Version();

        [DllImport("tensorflow.dll")]
        private extern static TF_Status TF_NewStatus();

        [DllImport("tensorflow.dll")]
        private extern static void TF_DeleteStatus(TF_Status status);

        [DllImport("tensorflow.dll")]
        private extern static TF_Code TF_GetCode(TF_Status status);

        [DllImport("tensorflow.dll")]
        private extern static CharPtr TF_Message(TF_Status status);

        [DllImport("tensorflow.dll")]
        private extern static TF_Buffer TF_NewBuffer();

        [DllImport("tensorflow.dll")]
        private extern static void TF_DeleteBuffer(TF_Buffer buffer);

        [DllImport("tensorflow.dll")]
        private extern static TF_Graph TF_NewGraph();

        [DllImport("tensorflow.dll")]
        private extern static void TF_DeleteGraph(TF_Graph graph);

        [DllImport("tensorflow.dll")]
        private extern static TF_ImportGraphDefOptions TF_NewImportGraphDefOptions();

        [DllImport("tensorflow.dll")]
        private extern static void TF_DeleteImportGraphDefOptions(TF_ImportGraphDefOptions options);

        [DllImport("tensorflow.dll")]
        private extern static void TF_GraphImportGraphDef(TF_Graph graph, TF_Buffer graphDef, TF_ImportGraphDefOptions options, TF_Status status);

        [DllImport("tensorflow.dll")]
        private extern static TF_SessionOptions TF_NewSessionOptions();

        [DllImport("tensorflow.dll")]
        private extern static void TF_DeleteSessionOptions(TF_SessionOptions options);

        [DllImport("tensorflow.dll")]
        private extern static TF_Session TF_NewSession(TF_Graph graph, TF_SessionOptions options, TF_Status status);

        [DllImport("tensorflow.dll")]
        private extern static void TF_CloseSession(TF_Session session, TF_Status status);

        [DllImport("tensorflow.dll")]
        private extern static void TF_DeleteSession(TF_Session session, TF_Status status);

        [DllImport("tensorflow.dll")]
        private extern static TF_Tensor TF_NewTensor(TF_DataType type, LongPtr dims, int numDims, VoidPtr data, int length, TensorDeallocator deallocator, VoidPtr arg);

        [DllImport("tensorflow.dll")]
        private extern static VoidPtr TF_TensorData(TF_Tensor tensor);

        [DllImport("tensorflow.dll")]
        private extern static void TF_DeleteTensor(TF_Tensor tensor);

        [DllImport("tensorflow.dll")]
        private extern static TF_Operation TF_GraphOperationByName(TF_Graph graph, CharPtr operationName);

        [DllImport("tensorflow.dll")]
        private extern static void TF_SessionRun(TF_Session session, TF_Buffer options, TF_Output inputs, ref TF_Tensor inputValues, int numInputs, TF_Output outputs, ref TF_Tensor outputValues, int numOutputs, ref TF_Operation operators, int numTargets, TF_Buffer metadata, TF_Status status);
    }
}
