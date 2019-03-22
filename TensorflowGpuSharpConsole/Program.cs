using RiceCakeSoftware.TensorflowGpuSharpConsole;

namespace TensorflowGpuSharpConsole
{
    class Program
    {
        private const int IMAGE_SIZE = 28;
        private const int IMAGE_BPP = 1;

        static void Main()
        {
            using (Tensorflow tensorflow = new Tensorflow())
            {
                System.Console.WriteLine($"{tensorflow.GetVersion()}");
                tensorflow.CreateSession("keras_fashion_mnist.pb");
                float[] pixels = new float[IMAGE_SIZE * IMAGE_SIZE * IMAGE_BPP];
                float[] probabries = tensorflow.RunSession(pixels, IMAGE_SIZE, IMAGE_SIZE, IMAGE_BPP, "conv2d_1_input", "dense_2/Softmax", 10);
                System.Console.WriteLine($"{string.Join(",", probabries)}");
            }
        }
    }
}
