using System;

namespace RiceCakeSoftware.TensorflowGpuSharpConsole
{
    public class TensorflowException : Exception
    {
        public TensorflowException(string message) : base(message) { }
    }
}
