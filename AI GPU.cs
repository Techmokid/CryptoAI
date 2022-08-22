using System;
using System.IO;
using System.Numerics;
using System.Threading;
using System.Diagnostics;
using System.Collections.Generic;

using ComputeSharp;
using Newtonsoft.Json;

using NI = CryptoAI.NetworkInterface;

namespace CryptoAI {
	// The indexing for the nodes array:
	//  - Genome_GPU[i].Nodes_Start_Index   gives us where to start in the index array.
	//  - Genome_GPU[i+1].Nodes_Start_Index gives us where to end   in the index array.
	//  - The nodes are ordered in genome 1, genome 2, genome 3, etc with data positions listed above
	//  - In each genome, the nodes in the nodes array is ordered "Starting nodes", "Hidden nodes", "Output nodes"
	
	// So, to get the outputs of a genome, we:
	// - Get the genome index from the thread.x
	// - Read out the "Nodes_Start_Index" variable in the genome to figure out where to start reading the node data from
	// - Set each of the node inputs from "Nodes_Start_Index" all the way up until the nodes starting showing (nII==False)
	// - Now iterate forward in memory until we get (nIO==True) and put this in index variable integer "OutputNodesStartIndex"
	// - For each node between "OutputNodesStartIndex" and "Genome_GPU[i+1].Nodes_Start_Index - 1", we ask it for it's output
	
	// To get a nodes output:
	// - for (int i = nodes[nodeIndex].weightIndexes; i < nodes[nodeIndex+1].weightIndexes; i++)
	// - Now that we are iterating over every connection, we simply call the node in nodes[nodeConnections[i].NodeID] and do the algorithm
	
	public partial class AI_GPU {
		public static Network_GPU NGPU;
		public static string saveDirectory = "";
		public static int numberOfIndexesPerThread = 10000;
		public static List<double[]> trainingData = null;
		public static double[] currentTrainingData = null;
		public static bool useGPU = true;
		static bool CPUIsReadyForCopying = false;
		static bool GPUIsReady = false;
		static bool once = false;
		static double trainingWanderVariable = 1.0;
		
		public struct Genome_GPU {
			public Genome_GPU() { }
			public int ID = -1;
			public double fitness = -99999;
			public double prev_fitness = -99999;
			public int Nodes_Start_Index = -1;
			public int Nodes_End_Index = -1;
		}
		
		public struct Node_GPU {
			public Node_GPU() { }
			public int ID = -1;			//Identification code		// The ID of this node
			
			public double nTT = 0;		//Node trigger typeof		// 0 is step, 1 is sigmoid
			public double nB = 0;		//Node bias
			public double pNB = 0;		//Previous node bias
			
			public bool nII = false; 	//node Is Input				// Is the node is an input or not
			public bool nIO = false; 	//node Is Output			// Is the node is an output or not
			public bool tNIO = false;	//node temporary output		// For some reason, nIO is not being read correctly on the GPU. This is the fix?
			public double nIV = 0;		//node Input Value			// If the node is an input, what have we entered
			public double pO = -99999;	//precalculated Output		// This variable just allows for quicker genome output computing
			
			public int wSI = 0; 		//weights Start Index		// This is the position in the weights array where the start of this nodes connections are held
			public int wEI = 0;			//weights End Index			// This is the position in the weights array where the end of this nodes connections are held
		}
		
		public struct NodeConnection_GPU {
			public int NodePos;
			public double Weight;
			public double Prev_Weight;
		}
		
		public struct Network_GPU {
			public Genome_GPU[] genomes;
			public Node_GPU[] nodes;
			public NodeConnection_GPU[] connections;
		}
		
		public void CreateTrainingSavingLoop() {
			CPUIsReadyForCopying = false;
			GPUIsReady = false;
			Thread workerThreadGPU = new Thread(CheckGPUStatus); 
			workerThreadGPU.Start();
			SaveNetworkGPU();
			CPUIsReadyForCopying = true;
			Log.Debug("GPU","Waiting for GPU to finish...");
			while(!GPUIsReady) {} //Do nothing
		}
		
		public void SaveNetworkGPU(string dir) { saveDirectory=dir + "/"; SaveNetworkGPU(); }
		public void SaveNetworkGPU() {
			// Start the GPU handling thread.
			// Simply just keep telling the GPU to do another "x" iterations of the network until the CPU has finished saving it's copy of the AI to disk
			if (saveDirectory == "") {
				Log.Error("CryptoAI","Could not save GPU network. SaveNetworkGPU was called without setting save directory");
				NI.quit();
			}
			
			string netSaveDir = saveDirectory + "Network Save Orig/";
			string netBackupDir = saveDirectory + "Network Save Backup/";
			string netStatusFile = saveDirectory + "saveStatus.txt";
			
			if (!Directory.Exists(saveDirectory))
				Directory.CreateDirectory(saveDirectory);
			
			//Save the original copy
			File.WriteAllText(netStatusFile,"1");
			if (Directory.Exists(netSaveDir)) {
				Log.Debug("CryptoAI","Clearing old primary save");
				Directory.Delete(netSaveDir,true);
				Log.Success("CryptoAI","Cleared old primary save");
			}
			Log.Debug("CryptoAI","Saving AI network - primary copy");
			Directory.CreateDirectory(netSaveDir);
			SaveGPUNetworkInternal(netSaveDir);
			Log.Success("CryptoAI","Saved AI network - primary copy");
			
			//Save the backup copy
			File.WriteAllText(netStatusFile,"2");
			if (Directory.Exists(netBackupDir)) {
				Log.Debug("CryptoAI","Clearing old backup save");
				Directory.Delete(netBackupDir,true);
				Log.Success("CryptoAI","Cleared old backup save");
			}
			Log.Debug("CryptoAI","Saving AI network - backup copy");
			NI.CopyFilesRecursively(netSaveDir,netBackupDir);
			Log.Success("CryptoAI","Saved AI network - backup copy");
			
			//Now finish up
			File.WriteAllText(netStatusFile,"0");
			Log.Success("CryptoAI","Saved AI network!");
		}
		
		public void SaveGPUNetworkInternal(string dir) {
			Stopwatch S = new Stopwatch();
			S.Start();
			Directory.CreateDirectory(dir + "Genomes/");
			Directory.CreateDirectory(dir + "Nodes/");
			Directory.CreateDirectory(dir + "Node Connections/");
			
			List<ThreadDataContainer4> TDC_List = new List<ThreadDataContainer4>();
			bool run = true;
			int genomeIndex = 0;
			int maxThreadCount = 16;
			int currentThreadCount = 0;
			
			Log.Print("AI Sace","Genome File Count: " + Math.Ceiling((float)NGPU.genomes.Length/(float)numberOfIndexesPerThread));
			Log.Print("AI Sace","Node File Count: " + Math.Ceiling((float)NGPU.nodes.Length/(float)numberOfIndexesPerThread));
			Log.Print("AI Sace","Node Connections File Count: " + Math.Ceiling((float)NGPU.connections.Length/(float)numberOfIndexesPerThread));
			
			//Now we actually start saving the AI to disk
			int numIterations = 0;
			while(run) {
				while (currentThreadCount < maxThreadCount) {
					if (numIterations == 20)
						once = true;
					numIterations++;
					
					if(genomeIndex == NGPU.connections.Length) { run = false; break; }
					
					//HERE WE WANT TO ACTUALLY START THE THREAD PROCESS
					ThreadDataContainer4 TDC = new ThreadDataContainer4();
					
					TDC.ID = genomeIndex;
					TDC.EndIndex = genomeIndex + Math.Min(numberOfIndexesPerThread - 1,NGPU.connections.Length-genomeIndex-1);
					
					//Console.WriteLine("Start: " + TDC.ID + "\t\tEnd: " + TDC.EndIndex + "\t\tDiff: " + (TDC.EndIndex - TDC.ID).ToString());
					//Console.WriteLine("Connections Start: " + genomeIndex.ToString());
					//Console.WriteLine("Connections End:   " + TDC.EndIndex.ToString());
					//Console.WriteLine("Connections Count: " + NGPU.connections.Length.ToString());
					
					if (TDC.EndIndex >= NGPU.connections.Length) {
						Log.Error("AI Save","Connection indexing error!!");
						NI.quit();
					}
					
					//TDC.path = dir + "Node Connections/Connection " + genomeIndex.ToString() + ".json";
					TDC.path = dir;
					TDC_List.Add(TDC);
					Thread workerThread = new Thread(SaveNetworkConnectionsGPU_MT); 
					workerThread.Start(TDC);
						
					currentThreadCount++;
					
					if (genomeIndex < NGPU.nodes.Length) {
						ThreadDataContainer4 TDC2 = new ThreadDataContainer4();
						
						TDC2.ID = genomeIndex;
						TDC2.EndIndex = genomeIndex + Math.Min(genomeIndex + numberOfIndexesPerThread - 1,NGPU.nodes.Length-genomeIndex-1);
						//TDC2.path = dir + "Node Connections/Connection " + genomeIndex.ToString() + ".json";
						TDC2.path = dir;
						//Console.WriteLine("Nodes Start: " + genomeIndex.ToString());
						//Console.WriteLine("Nodes End:   " + TDC2.EndIndex.ToString());
						//Console.WriteLine("Nodes Count: " + NGPU.nodes.Length.ToString());
						if (TDC2.EndIndex >= NGPU.nodes.Length) {
							Log.Error("AI Save","Node indexing error!!");
							NI.quit();
						}
						TDC_List.Add(TDC2);
						Thread workerThread2 = new Thread(SaveNetworkNodesGPU_MT); 
						workerThread2.Start(TDC2);
							
						currentThreadCount++;
					}
					
					if (genomeIndex < NGPU.genomes.Length) {
						ThreadDataContainer4 TDC3 = new ThreadDataContainer4();
						
						TDC3.ID = genomeIndex;
						TDC3.EndIndex = genomeIndex + Math.Min(genomeIndex + numberOfIndexesPerThread - 1,NGPU.genomes.Length-genomeIndex-1);
						//TDC3.path = dir + "Node Connections/Connection " + genomeIndex.ToString() + ".json";
						TDC3.path = dir;
						//Console.WriteLine("Genome Start: " + genomeIndex.ToString());
						//Console.WriteLine("Genome End:   " + TDC3.EndIndex.ToString());
						//Console.WriteLine("Genome Count: " + NGPU.genomes.Length.ToString());
						if (TDC3.EndIndex >= NGPU.genomes.Length) {
							Log.Error("AI Save","Genome indexing error!!");
							NI.quit();
						}
						TDC_List.Add(TDC3);
						Thread workerThread3 = new Thread(SaveNetworkGenomesGPU_MT); 
						workerThread3.Start(TDC3);
							
						currentThreadCount++;
					}
					
					genomeIndex += numberOfIndexesPerThread;
				}
				
				if (run) {
					for (int x = 0; x < TDC_List.Count; x++) {
						if (TDC_List[x].threadCompletionStatus) {
							TDC_List.RemoveAt(x);
							currentThreadCount--;
							x--;
						}
					}
				}
			}
			while(TDC_List.Count > 0) {
				for (int x = 0; x < TDC_List.Count; x++) {
					if (TDC_List[x].threadCompletionStatus) {
						TDC_List.RemoveAt(x);
						currentThreadCount--;
						x--;
					}
				}
			}
			
			S.Stop();
			Log.Print("AI Save","Actual save time taken: " + TimeFormatter((double)S.Elapsed.Milliseconds / 1000 / 60 / 60));
		}
		
		public void CheckGPUStatus(List<double[]> data) {
			if (trainingData != null) {
				trainingData = data;
				CheckGPUStatus();
			}
			GPUIsReady = true;
		}
		public void CheckGPUStatus() {
			if (trainingData != null) {
				while(!CPUIsReadyForCopying) {
					for(int i = 0; i < trainingData.Count; i++) {
						currentTrainingData = trainingData[i];
						TrainNetwork(100000);
						
						double worstFitness = double.PositiveInfinity;
						double bestFitness = double.NegativeInfinity;
						double meanFitness = 0;
						for (int x = 0; x < NGPU.genomes.Length; x++) {
							meanFitness += NGPU.genomes[x].fitness;
							if (NGPU.genomes[x].fitness > bestFitness)
								bestFitness = NGPU.genomes[x].fitness;
							if (NGPU.genomes[x].fitness < worstFitness)
								worstFitness = NGPU.genomes[x].fitness;
						}
						meanFitness /= NGPU.genomes.Length;
						Log.Debug("CPU",(i + 1) + "/" + trainingData.Count + "\t\tBest Fitness: " + bestFitness + "\t\tWorst Fitness: " + worstFitness + "\t\tMean Fitness: " + meanFitness);
					}
					if (!CPUIsReadyForCopying) {
						Log.Debug("CPU","CPU has not finished saving to disk. Iterating GPU again");
					} else {
						Log.Debug("CPU","CPU has finished saving to disk. Copying GPU AI to RAM");
					}
				}
			}
			GPUIsReady = true;
		}
		
		public void LoadNetworkGPU(string dir) { saveDirectory=dir + "/"; LoadNetworkGPU(); }
		public void LoadNetworkGPU() {
			if (saveDirectory == "") {
				Console.ForegroundColor = ConsoleColor.Red;
				Log.Error("CryptoAI","Could not load network. LoadNetwork was called without setting save directory");
				NI.quit();
			}
			
			Log.Warning("AI Config","Loading AI network can take several minutes...");
			
			string netSaveDir = saveDirectory + "Network Save Orig/";
			string netBackupDir = saveDirectory + "Network Save Backup/";
			string netStatusFile = saveDirectory + "saveStatus.txt";
			
			string loadDirectory = "";
			if (File.Exists(netStatusFile)) {
				//0 	Means it's completely safe to read from either.							Read from the first one
				//1 	Means it was in the middle of writing to the first copy. 				Read from the second
				//2 	Means it was in the middle of writing to the backup copy. 				Read from the first one
				//null  Means it was in the middle of writing to the save status directory.		Read from the first one
				try {
					int result = int.Parse(File.ReadAllText(netStatusFile));
					if (result == 0) {
						Log.Print("AI Config","Loading AI network from primary save");
						loadDirectory = netSaveDir;
					} else if (result == 1) {
						Log.Warning("AI Config","Primary save corrupted. Attempting file repair");
						NI.saveFileRepair();
						Log.Print("AI Config","Loading AI network from backup save");
						loadDirectory = netBackupDir;
					} else if (result == 2) {
						Log.Warning("AI Config","Backup save corrupted. Attempting file repair");
						NI.saveFileRepair();
						Log.Print("AI Config","Loading AI network from primary save");
						loadDirectory = netSaveDir;
					} else {
						Log.Error("AI Config","Invalid save status detected in save directory. Defaulting to primary save");
						NI.quit();
					}
				} catch {
					//null
					Log.Warning("AI Config","Error reading from save status file. Defaulting to primary save");
					loadDirectory = netSaveDir;
					File.WriteAllText(netStatusFile,"0");
					Log.Success("AI Config","Overwrote corrupt data in save file");
				}
			} else {
				Log.Warning("CryptoAI","Could not find save status file. Using default option");
				if (Directory.Exists(netSaveDir)) {
					Log.Print("AI Config","Loading AI network from primary save");
					loadDirectory = netSaveDir;
				} else if (Directory.Exists(netBackupDir)) {
					Log.Print("AI Config","Loading AI network from backup save");
					loadDirectory = netBackupDir;
				} else {
					Log.Error("CryptoAI","Could not load network. No save folder detected in directory");
					NI.quit();
				}
			}
			
			NGPU = new Network_GPU();
			
			string[] allGenomeDirectories = Directory.GetFiles(loadDirectory + "Genomes/");
			NGPU.genomes = new Genome_GPU[allGenomeDirectories.Length*numberOfIndexesPerThread];
			string[] allNodeDirectories = Directory.GetFiles(loadDirectory + "Nodes/");
			NGPU.nodes = new Node_GPU[allNodeDirectories.Length*numberOfIndexesPerThread];
			string[] allConnectionsDirectories = Directory.GetFiles(loadDirectory + "Node Connections/");
			NGPU.connections = new NodeConnection_GPU[allConnectionsDirectories.Length*numberOfIndexesPerThread];
			
			List<ThreadDataContainer4> TDC_List = new List<ThreadDataContainer4>();
			bool run = true;
			int genomeIndex = 0;
			int maxThreadCount = 16;
			int currentThreadCount = 0;
			while(run) {
				while (currentThreadCount < maxThreadCount) {
					if(genomeIndex == allConnectionsDirectories.Length) { run = false; break; }
				
					//HERE WE WANT TO ACTUALLY START THE THREAD PROCESS
					ThreadDataContainer4 TDC = new ThreadDataContainer4();
					
					TDC.ID = genomeIndex;
					TDC.path = allConnectionsDirectories[genomeIndex];
					TDC_List.Add(TDC);
					Thread workerThread = new Thread(LoadNetworkConnectionsGPU_MT); 
					workerThread.Start(TDC);
						
					currentThreadCount++;
					
					if (genomeIndex < allNodeDirectories.Length) {
						ThreadDataContainer4 TDC2 = new ThreadDataContainer4();
						
						TDC2.ID = genomeIndex;
						TDC2.path = allNodeDirectories[genomeIndex];
						TDC_List.Add(TDC2);
						Thread workerThread2 = new Thread(LoadNetworkNodesGPU_MT); 
						workerThread2.Start(TDC2);
							
						currentThreadCount++;
					}
					
					if (genomeIndex < allGenomeDirectories.Length) {
						ThreadDataContainer4 TDC3 = new ThreadDataContainer4();
						
						TDC3.ID = genomeIndex;
						TDC3.path = allGenomeDirectories[genomeIndex];
						TDC_List.Add(TDC3);
						Thread workerThread3 = new Thread(LoadNetworkGenomesGPU_MT); 
						workerThread3.Start(TDC3);
							
						currentThreadCount++;
					}
					
					genomeIndex++;
				}
				
				if (run) {
					for (int x = 0; x < TDC_List.Count; x++) {
						if (TDC_List[x].threadCompletionStatus) {
							TDC_List.RemoveAt(x);
							currentThreadCount--;
							x--;
						}
					}
				}
			}
			while(TDC_List.Count > 0) {
				for (int x = 0; x < TDC_List.Count; x++) {
					if (TDC_List[x].threadCompletionStatus) {
						TDC_List.RemoveAt(x);
						currentThreadCount--;
						x--;
					}
				}
			}
		}
		
		public void LoadNetworkGenomesGPU_MT(object data) {
			ThreadDataContainer4 TDC = (ThreadDataContainer4)data;
			Genome_GPU[] G = JsonConvert.DeserializeObject<Genome_GPU[]>(File.ReadAllText(TDC.path));
			for(int i = 0; i < G.Length; i++) {
				NGPU.genomes[G[i].ID] = G[i];
			}
			
			TDC.threadCompletionStatus = true;
		}
		
		public void LoadNetworkNodesGPU_MT(object data) {
			ThreadDataContainer4 TDC = (ThreadDataContainer4)data;
			Node_GPU[] N = JsonConvert.DeserializeObject<Node_GPU[]>(File.ReadAllText(TDC.path));
			for(int i = 0; i < N.Length; i++) {
				NGPU.nodes[N[i].ID] = N[i];
			}
			
			TDC.threadCompletionStatus = true;
		}
		
		public void LoadNetworkConnectionsGPU_MT(object data) {
			ThreadDataContainer4 TDC = (ThreadDataContainer4)data;
			NodeConnection_GPU[] C = JsonConvert.DeserializeObject<NodeConnection_GPU[]>(File.ReadAllText(TDC.path));
			for(int i = 0; i < C.Length; i++) {
				NGPU.connections[TDC.ID*numberOfIndexesPerThread + i] = C[i];
			}
			
			TDC.threadCompletionStatus = true;
		}
		
		public void SaveNetworkGenomesGPU_MT(object data) {
			ThreadDataContainer4 TDC = (ThreadDataContainer4)data;
			
			Genome_GPU[] G = new Genome_GPU[TDC.EndIndex - TDC.ID + 1];
			for(int i = TDC.ID; i <= TDC.EndIndex; i++) { G[i - TDC.ID] = NGPU.genomes[i]; }
			File.WriteAllText(TDC.path + "Genomes/Genome Data " + TDC.ID + ".json",JsonConvert.SerializeObject(G));
			
			TDC.threadCompletionStatus = true;
		}
		
		public void SaveNetworkNodesGPU_MT(object data) {
			ThreadDataContainer4 TDC = (ThreadDataContainer4)data;
			
			Node_GPU[] N = new Node_GPU[TDC.EndIndex - TDC.ID + 1];
			for(int i = TDC.ID; i <= TDC.EndIndex; i++) { N[i - TDC.ID] = NGPU.nodes[i]; }
			File.WriteAllText(TDC.path + "Nodes/Node Data " + TDC.ID + ".json",JsonConvert.SerializeObject(N));
			
			TDC.threadCompletionStatus = true;
		}
		
		public void SaveNetworkConnectionsGPU_MT(object data) {
			ThreadDataContainer4 TDC = (ThreadDataContainer4)data;
			Stopwatch s = new Stopwatch();
			
			bool doOnce = false;
			if (once) {
				doOnce = true;
				once = false;
				s.Start();
			}
			
			NodeConnection_GPU[] C = new NodeConnection_GPU[TDC.EndIndex - TDC.ID + 1];
			for(int i = TDC.ID; i <= TDC.EndIndex; i++) { C[i - TDC.ID] = NGPU.connections[i]; }
			File.WriteAllText(TDC.path + "Node Connections/Connection Data " + TDC.ID + ".json",JsonConvert.SerializeObject(C));
			
			if (doOnce) {
				s.Stop();
				Log.Debug("AI Thread", "Time per node connection: " + s.Elapsed.Milliseconds.ToString() + " ms");
				Log.Debug("AI Thread", "Time to save entire network: " + TimeFormatter((double)s.Elapsed.Milliseconds * NGPU.connections.Length / (double)numberOfIndexesPerThread / 1000 / 60 / 60));
			}
			
			
			TDC.threadCompletionStatus = true;
		}
		
		public string TimeFormatter(double x) {
			if (x > 24*365.25*1000) {
				return (x/24/365.25/1000).ToString() + " millenia";
			} else if (x >= 24*365.25*100) {
				return (x/24/365.25/100).ToString() + " centuries";
			} else if (x >= 24*365.25*10) {
				return (x/24/365.25/10).ToString() + " decades";
			} else if (x >= 24*365.25) {
				return (x/24/365.25).ToString() + " years";
			} else if (x >= 24) {
				return (x/24).ToString() + " days";
			} else if (x >= 1) {
				return (x/24).ToString() + " hours";
			} else if (x >= 1/60) {
				return (x*60).ToString() + " minutes";
			} else if (x >= 1/60/60) {
				return (x*60*60).ToString() + " seconds";
			} else if (x >= 1/60/60/1000) {
				return (x*60*60*1000).ToString() + " milliseconds";
			} else if (x >= 1/60/60/1000/1000) {
				return (x*60*60*1000*1000).ToString() + " microseconds";
			} else if (x >= 1/60/60/1000/1000/1000) {
				return (x*60*60*1000*1000*1000).ToString() + " nanoseconds";
			} else if (x >= 1/60/60/1000/1000/1000/1000) {
				return (x*60*60*1000*1000*1000*1000).ToString() + " picoseconds";
			} else {
				return "Time too small for formatting";
			}
		}
		
		public void TrainNetwork(int iterations) {
			int inputsCount = 0;
			int outputsCount = 0;
			for(int i = NGPU.genomes[0].Nodes_Start_Index; i <= NGPU.genomes[0].Nodes_End_Index; i++) {
				if (NGPU.nodes[i].nIO)
					outputsCount++;
				if (NGPU.nodes[i].nII)
					inputsCount++;
				if (NGPU.nodes[i].nII && NGPU.nodes[i].nIO)
					Log.Error("","NETWORK MALFUNCTION! Network detected duplicate input/output topology!");
			}
			
			outputsCount *= NGPU.genomes.Length;
			
			double[] inputsArray = new double[inputsCount];
			double[] outputsArray = new double[outputsCount];
			
			using ReadWriteBuffer<Genome_GPU> gBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(NGPU.genomes);
			using ReadWriteBuffer<Node_GPU> nBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(NGPU.nodes);
			using ReadWriteBuffer<NodeConnection_GPU> cBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(NGPU.connections);
			
			// For training data, the AI uses "inputsBuff" as a counter for the number of input nodes.
			// For standard network output, the AI uses "inputsBuff" as the network inputs to analyze.
			using ReadWriteBuffer<double> inputsBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(inputsArray);
			using ReadWriteBuffer<double> outputsBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(outputsArray);
			
			using ReadWriteBuffer<int> randOffsetBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(new int[NGPU.genomes.Length]);
			
			using ReadWriteBuffer<double> trainingBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(currentTrainingData);
			
			GraphicsDevice.Default.For(NGPU.nodes.Length, new GPU_AI_PerNode(
				gBuff,					// Genomes buffer
				nBuff,					// Nodes buffer
				cBuff,					// Node connections buffer
				inputsBuff,				// Inputs buffer
				outputsBuff,			// Outputs buffer
				iterations,				// The number of times to iterate training
				(int)Math.Floor(1000000*AI_Internal_Core.getRandomFloat()),	//Just a random seed for the AI
				NGPU.nodes.Length,		// The thread count (Technically there should be a variable for this, but idk what it is)
				randOffsetBuff,			// Random offsets buffer
				trainingBuff,			// The training data buffer
				true					// Are we training or just getting an output
			));
			
			gBuff.CopyTo(NGPU.genomes);
			nBuff.CopyTo(NGPU.nodes);
			cBuff.CopyTo(NGPU.connections);
		}
		
		public List<double[]> GetNetworkOutput(double[] inputs) {
			int outputsCount = 0;
			//for(int i = NGPU.genomes[0].Nodes_Start_Index; i <= NGPU.genomes[0].Nodes_End_Index; i++) {
			
			for(int i = 0; i < NGPU.nodes.Length; i++) {
				if (NGPU.nodes[i].nIO) {
					outputsCount++;
					NGPU.nodes[i].tNIO = true;
				}
				if (NGPU.nodes[i].nII && NGPU.nodes[i].nIO)
					Log.Error("","NETWORK ERROR! Network detected duplicate input/output node");
			}
			
			outputsCount *= NGPU.genomes.Length;
			
			Console.WriteLine("Pre Outputs: " + outputsCount);
			double[] outputsArray = new double[outputsCount];
			
			if (useGPU) {
				using ReadWriteBuffer<Genome_GPU> gBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(NGPU.genomes);
				using ReadWriteBuffer<Node_GPU> nBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(NGPU.nodes);
				using ReadWriteBuffer<NodeConnection_GPU> cBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(NGPU.connections);
				
				// For training data, the AI uses "inputsBuff" as a counter for the number of input nodes.
				// For standard network output, the AI uses "inputsBuff" as the network inputs to analyze.
				using ReadWriteBuffer<double> inputsBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(inputs);
				using ReadWriteBuffer<double> outputsBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(outputsArray);
				
				using ReadWriteBuffer<int> randOffsetBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(new int[NGPU.nodes.Length]);
				
				double[] tempTrainingData = new double[300];
				tempTrainingData[299] = 5;
				using ReadWriteBuffer<double> trainingBuff = GraphicsDevice.Default.AllocateReadWriteBuffer(tempTrainingData);
				
				GraphicsDevice.Default.For(NGPU.nodes.Length, new GPU_AI_PerNode(
					gBuff,
					nBuff,
					cBuff,
					inputsBuff,
					outputsBuff,
					1,
					(int)Math.Floor(1000000*AI_Internal_Core.getRandomFloat()),
					NGPU.nodes.Length,
					randOffsetBuff,
					trainingBuff,
					false
				));
				
				NGPU.nodes = new Node_GPU[NGPU.nodes.Length];
				nBuff.CopyTo(NGPU.nodes);
				for(int i = 0; i < NGPU.nodes.Length; i++) {
					//if (NGPU.nodes[i].nIO)
					//	Console.WriteLine("Output node: " + NGPU.nodes[i].ID);
					//if (NGPU.nodes[i].nII)
					//	Console.WriteLine("Input node: " + NGPU.nodes[i].ID);
					if (NGPU.nodes[i].nII && NGPU.nodes[i].tNIO && NGPU.nodes[i].nIO)
						Log.Error("","NETWORK ERROR 2! Network detected duplicate input/output node");
				}
				
				outputsBuff.CopyTo(outputsArray);
				//for(int i = 0; i < outputsArray.Length; i++) {
				for(int i = 0; i < 10; i++) {
					Console.WriteLine("Output: " + outputsArray[i]);
				}
			} else {
				// Use the CPU instead of the GPU for calculating the output of the network
				
			}
			
			//Split the outputsBuff array into a list of smaller "double" arrays
			//genome.Nodes_End_Index - genome.Nodes_Start_Index
			List<double[]> result = new List<double[]>();
			for (int g = 0; g < NGPU.genomes.Length; g++) {
				int outputsPerGenome = outputsArray.Length / NGPU.genomes.Length;
				double[] resultTemp = new double[outputsPerGenome];
				for(int i = 0; i < outputsPerGenome; i++) {
					resultTemp[i] = outputsArray[g*outputsPerGenome + i];
				}
				result.Add(resultTemp);
			}
			
			return result;
		}
		
		public void Convert_CPU_Network_To_GPU_Network(Network CPU_Net) {
			throw new Exception("CPU to GPU conversion no longer valid!");
			
			//Set up the array variables
			int nodeCount = 0;
			int connectionsCount = 0;
			foreach(Genome G in CPU_Net.genomes) {
				nodeCount += G.nodes.Length;
				foreach(Node N in G.nodes) {
					if (!N.nII) {
						connectionsCount += N.cN.Length;
					}
				}
			}
			
			NGPU.genomes = new Genome_GPU[CPU_Net.genomes.Length];
			NGPU.nodes = new Node_GPU[nodeCount];
			NGPU.connections = new NodeConnection_GPU[connectionsCount];
			
			//Now here we actually fill in the arrays
			int nodeStartIndex = 0;
			int connectionsStartIndex = 0;
			for (int g = 0; g < CPU_Net.genomes.Length; g++) {
				Genome_GPU G = new Genome_GPU();
				G.ID = CPU_Net.genomes[g].ID;
				G.Nodes_Start_Index = nodeStartIndex;
				G.Nodes_End_Index = nodeStartIndex + CPU_Net.genomes[g].nodes.Length - 1;
				NGPU.genomes[g] = G;
				
				//Now that we have the genome array filled in, we fill in the node array
				for (int n = 0; n < CPU_Net.genomes[g].nodes.Length; n++) {
					Node_GPU N = new Node_GPU();
					string tmpStr = CPU_Net.genomes[g].nodes[n].NodeActivationType.ToLower();
					if (tmpStr == "step")
						N.nTT = 0;
					else if (tmpStr == "sigmoid")
						N.nTT = 1;
					else
						N.nTT = -1;
					N.ID = 					CPU_Net.genomes[g].nodes[n].ID;
					N.nB = 					AI_Internal_Core.getRandomFloat();
					N.pNB =					N.nB;
					N.nII = 				CPU_Net.genomes[g].nodes[n].nII;
					N.nIO = 				CPU_Net.genomes[g].nodes[n].nIO;
					N.nIV = 				CPU_Net.genomes[g].nodes[n].nIV;
					N.pO = 					-99999;
					if (CPU_Net.genomes[g].nodes[n].nII) {
						N.wSI = -1;
						N.wEI = -1;
					} else {
						N.wSI = 	connectionsStartIndex;
						N.wEI = 	connectionsStartIndex + CPU_Net.genomes[g].nodes[n].cN.Length - 1;
						
						//Now that we have the node array sorted out, we can just do the node connections array
						for (int nC = 0; nC < CPU_Net.genomes[g].nodes[n].cN.Length; nC++) {
							NodeConnection_GPU NC = new NodeConnection_GPU();
							NC.NodePos = nodeStartIndex + CPU_Net.genomes[g].nodes[n].cN[nC].n.ID;
							NC.Weight = CPU_Net.genomes[g].nodes[n].cN[nC].w;
							NGPU.connections[nC + connectionsStartIndex] = NC;
						}
						
						connectionsStartIndex += CPU_Net.genomes[g].nodes[n].cN.Length;
					}
					
					NGPU.nodes[nodeStartIndex + N.ID] = N;
				}
				
				nodeStartIndex += CPU_Net.genomes[g].nodes.Length;
			}
			
			//NGPU.prev_connections = (NodeConnection_GPU[])NGPU.connections.Clone();	// BE CAREFUL HERE!! We want to clone a copy of the array, not create identical REFERENCES to the array
		}
		
		public void NewNetworkGPU(int genomeCount, int inputNodes, int nodesPerLayer, int hiddenLayerCount, int outputNodes) {
			int nodeCountPerGenome = inputNodes + nodesPerLayer*hiddenLayerCount + outputNodes;
			int connCountPerGenome = inputNodes*nodesPerLayer + hiddenLayerCount*(hiddenLayerCount-1) + outputNodes*nodesPerLayer;
			
			Log.Debug("AI GPU","New GPU Network Data:");
			Log.Debug("AI GPU"," - Genome Count: " + genomeCount.ToString());
			Log.Debug("AI GPU"," - Nodes Per Genome: " + nodeCountPerGenome.ToString());
			Log.Debug("AI GPU"," - Node Connections Per Genome: " + connCountPerGenome.ToString());
			Log.Debug("AI GPU"," - Nodes Total: " + (nodeCountPerGenome*genomeCount).ToString());
			Log.Debug("AI GPU"," - Connections Total: " + (connCountPerGenome*genomeCount).ToString());
			//Log.Debug("AI GPU","");
			//Log.Debug("AI GPU"," - Genome Folder Size: " + ((double)genomeCount * (double)Marshal.SizeOf(new Genome_GPU())/1000000).ToString() + " MB");
			//Log.Debug("AI GPU"," - Node Folder Size: " + ((double)nodeCountPerGenome*(double)genomeCount * (double)Marshal.SizeOf(new Node_GPU())/1000000).ToString() + " MB");
			//Log.Debug("AI GPU"," - Node Connections Folder Size: " + ((double)connCountPerGenome*(double)genomeCount * (double)Marshal.SizeOf(new NodeConnection_GPU())/1000000).ToString() + " MB");
			
			NGPU = new Network_GPU();
			int totalNodeCountPerGenome = (inputNodes +
				nodesPerLayer*hiddenLayerCount +
				outputNodes);
			int totalNodeConnectionsCountPerGenome = (inputNodes*nodesPerLayer + 
				nodesPerLayer*nodesPerLayer*hiddenLayerCount + 
				outputNodes*nodesPerLayer) * genomeCount;
			
			NGPU.genomes = new Genome_GPU[genomeCount];
			for(int i = 0; i < NGPU.genomes.Length; i++) {
				NGPU.genomes[i].ID = i;
				NGPU.genomes[i].fitness = double.NegativeInfinity;
				NGPU.genomes[i].prev_fitness = double.NegativeInfinity;
				NGPU.genomes[i].Nodes_Start_Index = totalNodeCountPerGenome * i;
				NGPU.genomes[i].Nodes_End_Index = totalNodeCountPerGenome * (i+1) - 1;
			}
			
			//public double tT = 0;								// Minimum amount of input before triggering
			//public double pTT = 0;
			//public double tS = 0;								// Strength of the output transmition
			//public double pTS = 0;							// 
			//public bool nII = false; 							// Is the node is an input or not
			//public bool nIO = false; 							// Is the node is an output or not
			//public int wSI = 0; 					// Complex. This is the position in the array where the weights are held
			//public int wEI = 0;
			
			NGPU.connections = new NodeConnection_GPU[totalNodeConnectionsCountPerGenome];
			
			AI_Internal_Core.getRandomFloat();
			int tempWeightsCount = 0;
			NGPU.nodes = new Node_GPU[totalNodeCountPerGenome * genomeCount];
			for(int i = 0; i < NGPU.nodes.Length; i++) {
				NGPU.nodes[i].ID = i;
				
				//char[] genomeActivationTypes = new char[] { 'c','n' };
				//int randomIndex = AI_Internal_Core.rand.Next(genomeActivationTypes.Length);
				//NGPU.nodes[i].nAT = genomeActivationTypes[randomIndex];
				NGPU.nodes[i].nTT = 1;
				
				NGPU.nodes[i].nB = AI_Internal_Core.getRandomFloat();	//Node Bias
				NGPU.nodes[i].pNB = NGPU.nodes[i].pNB;
				
				int startOfGenome = (int)(totalNodeCountPerGenome * Math.Floor((double)NGPU.nodes[i].ID / (double)totalNodeCountPerGenome));
				NGPU.nodes[i].nII = i < startOfGenome + inputNodes;
				NGPU.nodes[i].nIO = i >= startOfGenome + inputNodes + nodesPerLayer*hiddenLayerCount;
				
				//NGPU.nodes[i].nIV = 0;
				//NGPU.nodes[i].pO = -99999;
				NGPU.nodes[i].wSI = tempWeightsCount;
				
				int startOfNonInputs = startOfGenome + inputNodes;
				int startOfLastLayer = (int)(startOfNonInputs + Math.Floor((double)(i - startOfNonInputs)/(double)nodesPerLayer) - nodesPerLayer);
				int numConnections = nodesPerLayer;
				if (NGPU.nodes[i].nIO) {
					NGPU.nodes[i].wEI = tempWeightsCount + numConnections - 1;
					
					int firstIndex = startOfGenome + totalNodeCountPerGenome - outputNodes - nodesPerLayer;
					for (int x = 0; x < numConnections; x++) {
						int pos = tempWeightsCount + x;
						NGPU.connections[pos].NodePos = firstIndex + x;
						NGPU.connections[pos].Weight = AI_Internal_Core.getRandomFloat();
						NGPU.connections[pos].Prev_Weight = NGPU.connections[pos].Weight;
					}
					
					tempWeightsCount += numConnections;
				} else if (!NGPU.nodes[i].nII) {
					bool isSecondLayer = i < startOfGenome + inputNodes + nodesPerLayer;
					if (isSecondLayer) { numConnections = inputNodes; }
					NGPU.nodes[i].wEI = tempWeightsCount + numConnections - 1;
					tempWeightsCount += numConnections;
					
					int startPoint = startOfLastLayer;
					if (isSecondLayer)
						startPoint = startOfGenome;
					for (int x = 0; x < numConnections; x++) {
						int pos = NGPU.nodes[i].wSI + x;
						NGPU.connections[pos].NodePos = startPoint + x;
						NGPU.connections[pos].Weight = AI_Internal_Core.getRandomFloat();
						NGPU.connections[pos].Prev_Weight = NGPU.connections[pos].Weight;
					}
				}
			}
			
			TrimNetwork();
		}
		
		public void TrimNetwork() {
			int trimmableGenomes = 0;
			for (int i = 0; i < NGPU.genomes.Length; i++) {
				if (NGPU.genomes[i].ID == -1)
					trimmableGenomes++;
			}
			
			int trimmableNodes = 0;
			for (int i = 0; i < NGPU.nodes.Length; i++) {
				if (NGPU.nodes[i].ID == -1)
					trimmableNodes++;
			}
			
			//Console.WriteLine("Trimmable Genomes: 		" + trimmableGenomes.ToString());
			//Console.WriteLine("Trimmable Nodes: 		" + trimmableNodes.ToString());
			//Console.WriteLine("Trimmable Connections: 	" + trimmableConnections.ToString());
			
			if (trimmableGenomes > 0)
				throw new Exception(trimmableGenomes + " genomes found with unset ID");
			if (trimmableNodes > 0)
				throw new Exception(trimmableNodes + " nodes found with unset ID");
		}
		
		[AutoConstructor]
		public readonly partial struct GPU_AI_PerNode : IComputeShader {
			public readonly ReadWriteBuffer<Genome_GPU> genomes;	             // This contains the genome objects that tell the GPU where each genome starts
			public readonly ReadWriteBuffer<Node_GPU> nodes;		             // This contains the node objects
			public readonly ReadWriteBuffer<NodeConnection_GPU> nodeConnections; // This contains all of the node connections data
			
			public readonly ReadWriteBuffer<double> genomeInputs;
			public readonly ReadWriteBuffer<double> genomeOutputs;
			
			public readonly int numberOfIterations;								 // How many times do we want to iterate the AI before handing back to the CPU
			public readonly int randomSeed;
			public readonly int maxThreadCount;
			
			public readonly ReadWriteBuffer<int> randomOffset;
			
			public readonly ReadWriteBuffer<double> trainingDataArray;
			public readonly bool isTraining;
			
			public void Execute() {
				if (isTraining == false) {
					CalculateNodeOutput(ThreadIds.X);
					return;
				}
				return;
				int randomMarketDataTicker = 0;
				double fakeCoins = 0;
				double fakeWallet = 1000;
				double fakeTotal = 1000;
				double networkBuyTax = 0.01;
				double networkSellTax = 0.01;
				int genomeInputsStart = genomes[GetGenomeIndex(ThreadIds.X)].Nodes_Start_Index;
				int outputsPerGenome = genomeOutputs.Length/genomes.Length;
				for(int iteration = 0; iteration < numberOfIterations; iteration++) {
					for(int trainingDataIndex = 0; trainingDataIndex < trainingDataArray.Length - genomeInputs.Length; trainingDataIndex++) {
						if (nodes[ThreadIds.X].nII) {
							//Set the training data to the node input here
							nodes[ThreadIds.X].nIV = trainingDataArray[trainingDataIndex + ThreadIds.X - genomeInputsStart];
							nodes[ThreadIds.X].pO = 1;//nodes[ThreadIds.X].nIV;
						}
						
						//Wait until all the input nodes are set.
						if (!nodes[ThreadIds.X].nII) {
							for (int n = 0; n < genomeInputs.Length; n++) {
								while(nodes[genomeInputsStart + n].pO == -99999) {}		
							}
							
							//Now that all the input nodes are set, we can run the calculation
							CalculateNodeOutput(ThreadIds.X);
						} else {
							for (int n = 0; n < outputsPerGenome; n++) {
								while(nodes[genomeInputsStart + n].pO == -99999) {}		
							}
						}
						
						//One single node per genome gets the priveledge to calculate the fitness
						if (ThreadIds.X == genomeInputsStart) {
							int outputPos = GetGenomeIndex(ThreadIds.X) * outputsPerGenome;
							if ((genomeOutputs[outputPos] > 0) && (fakeCoins > 0)) {
								fakeWallet = fakeCoins * trainingDataArray[trainingDataIndex] * (1 - networkSellTax);		//Sell
								fakeCoins = 0;
							}
							
							if ((genomeOutputs[outputPos + 1] > 0) && (fakeWallet > 0)) {
								fakeCoins = fakeWallet / trainingDataArray[trainingDataIndex];		//Buy
								fakeWallet = 0;
							}
							
							randomMarketDataTicker++;
							UpdateFakeMarketData(randomMarketDataTicker, ThreadIds.X);
						}
						
						//Synchronise all threads so that the nodes don't start calculating asychronously and causing havoc in memory
						nodes[ThreadIds.X].pO = -99999;
						for (int i = 0; i < genomeOutputs.Length/genomes.Length; i++) {
							while(nodes[genomeInputsStart + i].pO != -99999) {}
						}
						
						fakeTotal += fakeWallet + fakeCoins*trainingDataArray[trainingDataIndex + genomeInputs.Length];
					}
					
					//genomes[GetGenomeIndex(ThreadIds.X)].fitness = genomeOutputs[GetGenomeIndex(ThreadIds.X) * outputsPerGenome];//fakeTotal;
					genomes[GetGenomeIndex(ThreadIds.X)].fitness = fakeTotal;
					RollbackAndAdjustWeight(ThreadIds.X);
				}
			}
			
			void UpdateFakeMarketData(int ticker, int ID) {
				for (int i = 0; i < trainingDataArray.Length - 1; i++) {
					trainingDataArray[i] = trainingDataArray[i + 1];
				}
				
				// This follows a specific type of random number generation that I've proven via python works
				double oldPrice = trainingDataArray[trainingDataArray.Length-1];
				double MIN_PRICE = 0.1;
				double MAX_PRICE = 9.9;
				
				double volatility = 8;
				double changePercent = 2 * volatility * GetRandomFloat(ID);
				
				if (changePercent > volatility)
					changePercent -= (2 * volatility);
				
				double changeAmount = oldPrice * changePercent/100;
				double newPrice = oldPrice + changeAmount;
				
				if (newPrice < MIN_PRICE)
					newPrice += Math.Abs(changeAmount) * 2;
				else if (newPrice > MAX_PRICE)
					newPrice -= Math.Abs(changeAmount) * 2;
				
				trainingDataArray[trainingDataArray.Length-1] = newPrice;
			}
			
			double Exp(double x) { return (double)Math.Pow(Math.E,(float)x); }
			
			void CalculateNodeOutput(int ID) {
				int genomePos = GetGenomeIndex(ID);
				
				//If we are an input node, take the value from the inputs array
				if (nodes[ID].nII) {
					int i = ID - genomes[genomePos].Nodes_Start_Index;
					nodes[ID].nIV = genomeInputs[i];
					nodes[ID].pO = genomeInputs[i];
					return;
				}
				return;
				//If we are not an input node, we have to calculate our output value based on the previous nodes
				//All nodes are doing this simultaneously, so we just wait a moment until the nodes before us have finished, then we can give a response
				//Yes yes, I know that pausing execution and waiting for something on the GPU is bad. This is my code, and if it works, it can't be that bad
				double result = 0;
				for (int i = nodes[ID].wSI; i <= nodes[ID].wEI; i++) {
					int prevNodeIndex = nodeConnections[i].NodePos;
					//while (nodes[prevNodeIndex].pO == -99999) {}
					result += nodes[prevNodeIndex].pO * nodeConnections[i].Weight;
				}
				
				if (nodes[ID].nTT == 0)
					nodes[ID].pO = (result > 0) ? 0 : 1;  						// Step Function
				else if (nodes[ID].nTT == 1)
					nodes[ID].pO = 1 / (1 + Exp(-result)); 						// Sigmoid
				else if (nodes[ID].nTT == 2)
					nodes[ID].pO = (Exp(2*result) - 1) / (Exp(2 * result) + 1); // Hyperbolic Tangent
				else if (nodes[ID].nTT == 3)
					nodes[ID].pO = result / (1 + Exp(-result)); 				// SiLU
				else if (nodes[ID].nTT == 4)
					nodes[ID].pO = Math.Max(0, result); 						// ReLU
				
				//Is the node an output? If so, we must put our result into the output array of the GPU
				if (nodes[ID].tNIO) {
					int outputsPerGenome = genomeOutputs.Length/genomes.Length;
					//int outputArrayStartPos = outputsPerGenome*genomePos;
					//int outputArrayEndPos = (outputsPerGenome + 1)*genomePos - 1;
					int genomeNodesEnd = genomes[genomePos].Nodes_End_Index;
					int outputNodesIndexStart = genomeNodesEnd - outputsPerGenome + 1;
					int outputNodeIndex = outputsPerGenome*genomePos + ID - outputNodesIndexStart;
					
					genomeOutputs[outputsPerGenome] = nodes[ID].pO;
				}
				
			}
			
			void RollbackAndAdjustWeight(int nodeID) {
				int genIndex = GetGenomeIndex(nodeID);
				if (genomes[genIndex].fitness < genomes[genIndex].prev_fitness) {
					nodes[nodeID].nB = nodes[nodeID].pNB;
					for (int i = nodes[nodeID].wSI; i <= nodes[nodeID].wEI; i++) {
						nodeConnections[i].Weight = nodeConnections[i].Prev_Weight;
					}
				}
				
				nodes[nodeID].pO = -99999;
				
				double adjustmentVar = 0.05;
				nodes[nodeID].pNB = nodes[nodeID].nB;
				nodes[nodeID].nB += GetRandomFloat(nodeID) * adjustmentVar * 2 - adjustmentVar;
				for (int i = nodes[nodeID].wSI; i <= nodes[nodeID].wEI; i++) {
					nodeConnections[i].Prev_Weight = nodeConnections[i].Weight;
					nodeConnections[i].Weight = clamp(
						nodeConnections[i].Weight + (
							GetRandomFloat(nodeID) * adjustmentVar * 2 - adjustmentVar
						),-1,1
					);
				}
			}
			
			int GetGenomeIndex(int nodeID) { return (int)Math.Floor((float)((double)genomes.Length * (double)nodeID / (double)nodes.Length)); }
			
			double clamp(double x, double m, double M) {
				if (x > M)
					x = M;
				if (x < m)
					x = m;
				return x;
			}
			
			// Simple code for getting a random number on the GPU
			double dot(double a1, double a2, double b1, double b2) { return a1*b1 + a2*b2; }
			double frac(double x) { return  x - Math.Floor((float)x); }
			double GetRandomFloat(int nodeID) {
				int randomInputSeed = nodeID + randomOffset[nodeID] * maxThreadCount + randomSeed;
				randomOffset[nodeID] = randomOffset[nodeID] + 1;
				
				double vec1A = (double)randomInputSeed;
				double vec1B = 3.072 * (double)randomInputSeed + 0.518;
				double vec2A = 12.9898;
				double vec2B = 78.233;
				
				return frac(
					Math.Sin(
						(float)dot(vec1A, vec1B, vec2A, vec2B)
					) * 43758.5453
				);
			}
		}
	}
	
	public class ThreadDataContainer4 {
		public bool threadCompletionStatus = false;
		public int ID;
		public string path;
		public int EndIndex;
	}
}









