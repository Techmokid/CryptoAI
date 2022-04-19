using System.Threading;
using System.Collections.Generic;

using ComputeSharp;

namespace BasicAlgorithm {
	public partial class CryptoAlgorithm {
		public static List<double[]> trainingData = null;
		public static List<string[]> result = new List<string[]>();
		public static int dataAnalyticsViewWidth = 200;
		public static int minimumDataAnalysisLoops = 50;
		
		
		public void analyzeData(List<double[]> data) { trainingData = data; analyzeData(); }
		public void analyzeData() {
			//TDC.ID = genomeIndex;
			//TDC.data = trainingData[genomeIndex];
		}
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		public struct DataIndexObject {
			public int startIndex;
			public int endIndex;
		}
		
		[AutoConstructor]
		public readonly partial struct GPU_Algorithm : IComputeShader {
			public readonly ReadWriteBuffer<DataIndexObject> dataIndexes;
			public readonly ReadWriteBuffer<double> data;
			
			public void Execute() {
				int ID = ThreadIds.X;
				dataIndexes[ID].startIndex = 0;
			}
			
		//	public void CalculateOutput() {
		//		int startIndex = 0;
		//		int endIndex = 0;
		//		
		//		//smoothInputData(inputData,4,startIndex,endIndex);
		//
		//		//for(int iteration = 0; iteration < inputData.Length - dataAnalyticsViewWidth; iteration++) {
		//		//	for(int i = 0; i < dataAnalyticsViewWidth; i++) {
		//		//		//Here we iterate through every 
		//		//	}
		//		//}
		//	}
		//	
		//	public double[] smoothInputData(double[] x, int smoothingVal, int startIndex, int endIndex) {
		//		return x;
		//	}
		}
	}
}