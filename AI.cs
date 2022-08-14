using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Diagnostics;
using System.Collections.Generic;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace CryptoAI {
	public static class AI_Internal_Core {
		public static bool initialized = false;
        public static System.Random rand;
		
		public static Network n = null;
		
        public static void SetupRandom() { rand = new System.Random(); }
		public static double getRandomFloat() {
			if (!initialized)
				SetupRandom();
			initialized = true;
			
			return (double)rand.NextDouble();
		}
    }
	
	public struct NodeConnection {
		public Node n;
		public double w;
		public double prev_w = -99999;
		public NodeConnection(Node x, double y) {n=x;w=y;}
	}
	
	public class Node {
		public string NodeActivationType = "";
		public int ID = -1;									// The ID of this node
		public NodeConnection[] cN = new NodeConnection[0];	// All of the connected nodes downstream (Rolling end to start)
		
		public double tT = 0;								// Minimum amount of input before triggering
		public double prev_tT = 0;
		public double tS = 0;								// Strength of the output transmition
		public double prev_tS = 0;
		
		public bool nII = false; 							// Is the node is an input or not
		public bool nIO = false; 							// Is the node is an output or not
		public double nIV = 0;								// If the node is an input, what have we entered
		public double pO = -99999;							// This variable just allows for quicker genome output computing
		
		public Node(int _ID, NodeConnection[] _connected_nodes) {
			ID = _ID;
			cN = _connected_nodes;
		}
		
		public double getNodeOutput() {
			if (pO != -99999)
				return pO;
			
			if (nII)
				return nIV;
			
			double inVals = 0;
			for (int i = 0; i < cN.Length; i++) {
				inVals += cN[i].n.getNodeOutput() * cN[i].w;
			}
			
			switch (NodeActivationType.ToLower()) {
				case "neural":
					if (tT < (inVals/((double)cN.Length))) {
						//Enough of the inputs have triggered
						pO = tS;
						return tS;
					}
					pO = 0;
					return 0;
				case "classic":
					return inVals/((double)cN.Length);
				default:
					return 0;
			}
		}
		
		public void resetNodeToPreviousState() {
			pO = -99999;
			tT = prev_tT;
			tS = prev_tS;
			
			for(int i = 0; i < cN.Length; i++) {
				cN[i].w = cN[i].prev_w;
			}
		}
	}
	
	public class Genome {
		public Node[] inputNodes = new Node[0];
		public Node[] outputNodes = new Node[0];
		public Node[] nodes = new Node[0];
		
		// Classic is the way I built my first AI
		// Neural is the all-or-nothing-response
		public string GenomeType = "";
		
		public double fitness;
		public double prev_fitness;
		public int ID = 0;
		
		public void resetGenomeToPreviousState() {
			for(int i = 0; i < nodes.Length; i++) {
				nodes[i].resetNodeToPreviousState();
				nodes[i].NodeActivationType = GenomeType;
			}
		}
		
		public void resetPrecalculations() {
			for(int i = 0; i < nodes.Count(); i++) {
				nodes[i].pO = -99999;
				nodes[i].NodeActivationType = GenomeType;
			}
		}
		
		public void SetGenomeType(string type) {
			GenomeType = type;
			for(int i = 0; i < nodes.Count(); i++) {
				nodes[i].NodeActivationType = type;
			}
		}
		
		public void GetGenomeOutput(double[] inputValues, ref double[] outputValues) {
			if (inputNodes.Length != inputValues.Count()) {
				Log.Error("CryptoAI","Incorrect size of input array upon GetGenomeOutput()");
				Log.Error("CryptoAI","Received " + inputValues.Count().ToString() +
								  " values, but was expecting " + inputNodes.Count().ToString());
				NetworkInterface.quit();
			}
			
			for (int i = 0; i < inputNodes.Count(); i++) {
				inputNodes[i].nIV = inputValues[i];
			}
			
			for (int i = 0; i < outputValues.Count(); i++) {
				outputValues[i] = outputNodes[i].getNodeOutput();
			}
			
			return;
		}
	}
	
	public class Network {
		public Genome[] genomes = new Genome[0];
		public int inputNodesCount;
		public int outputNodesCount;
		public int hiddenLayerCount;
		public int hiddenLayerWidth;
	}
	
	public static class NetworkInterface {
		public static string saveDirectory = "";
		public static bool dumpAndPump = false;
		public static bool once = true;
		public static bool once2 = true;
		
		public static void SetDirectory(string dir) {
			dir = dir.Replace('\\','/');
			if (dir[dir.Length-1] != '/')
				dir += '/';
			
			saveDirectory = dir;
			//saveDirectory += "CryptoAI/";
			if (Directory.Exists(saveDirectory))
				Directory.CreateDirectory(saveDirectory);
		}
		
		public static Genome LoadBestGenome(string dir) { SetDirectory(dir); return LoadBestGenome(); }
		public static Genome LoadBestGenome() {
			if (saveDirectory == "") {
				Console.ForegroundColor = ConsoleColor.Red;
				Log.Error("CryptoAI","Could not load best genome. LoadBestGenome was called without setting save directory");
				quit();
			}
			
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
						Log.Print("AI Config","Loading best genome from primary save file");
						loadDirectory = netSaveDir;
					} else if (result == 1) {
						Log.Warning("AI Config","Primary save file corrupted. Loading best genome from backup save file");
						loadDirectory = netBackupDir;
					} else if (result == 2) {
						Log.Warning("AI Config","Backup save file corrupted. Attempting file repair");
						saveFileRepair();
						Log.Print("AI Config","Loading best genome from primary save file");
						loadDirectory = netSaveDir;
					} else {
						Log.Error("AI Config","Invalid save status detected in save directory");
						quit();
					}
				} catch {
					//null
					Log.Warning("AI Config","No save status detected. Loading best genome from primary save file");
					loadDirectory = netSaveDir;
				}
			} else {
				Log.Error("CryptoAI","Could not find save status file. Using default option");
				if (Directory.Exists(netSaveDir)) {
					Log.Print("AI Config","Loading best genome from primary save file");
					loadDirectory = netSaveDir;
				} else if (Directory.Exists(netBackupDir)) {
					Log.Print("AI Config","Loading best genome from backup save file");
					loadDirectory = netBackupDir;
				} else {
					Log.Error("CryptoAI","Could not load best genome. No save file detected in directory");
					quit();
				}
			}
			
			//Read from the text file to find which ID was best
			string bestIDDirectory = loadDirectory + "Best Genome.txt";
			if (!File.Exists(bestIDDirectory)) {
				Log.Error("CryptoAI","Could not load best genome. Could not locate best genome ID file at position: ");
				Log.Error("CryptoAI"," - " + bestIDDirectory);
				quit();
			}
			
			int bestID = int.Parse(File.ReadAllText(bestIDDirectory));
			return LoadGenome_CPU(loadDirectory + "Genome " + bestID.ToString() + ".json");
		}
		
		public static Genome LoadGenome_CPU(string dir) {
			Genome g = new Genome();
			
			if (!dumpAndPump) {
				g.GenomeType = File.ReadAllText(dir + "/Genome Activation Type.txt").Replace("\t","").Replace("\n","").Replace(" ","");
				
				string[] allNodeDirs = Directory.GetFiles(dir,"Node*.json");
				g.nodes = new Node[allNodeDirs.Length];
				for(int i = 0; i < g.nodes.Length; i++) {
					g.nodes[i] = new Node(i,null);
					g.nodes[i].ID = i;
				}
				
				//Just sort the node directories in ascending order
				for (int i = 0; i < allNodeDirs.Length; i++) {
					allNodeDirs[i] = dir + "/Node " + i.ToString() + ".json";
				}
				
				foreach (string nodeDir in allNodeDirs) {
					JObject nodeJSON = JObject.Parse(File.ReadAllText(nodeDir));
					int ID = (int)nodeJSON["ID"];
					
					Node n = new Node(ID,new NodeConnection[0]);
					//Does this node already exist in g.nodes? If so, overwrite "n"
					for(int i = 0; i < g.nodes.Length; i++) {
						if (g.nodes[i].ID == ID)
							n = g.nodes[i];
					}
					
					n.NodeActivationType = g.GenomeType;
					//n.NodeActivationType = (string)nodeJSON["nAT"];
					
					n.tT = (double)nodeJSON["tT"];
					n.tS = (double)nodeJSON["tS"];
					n.nII = (bool)nodeJSON["nII"];
					n.nIO = (bool)nodeJSON["nIO"];
					n.nIV = (double)nodeJSON["nIV"];
					n.pO = (double)nodeJSON["pO"];
					
					if (!n.nII) {
						List<NodeConnection> nC_List = new List<NodeConnection>();
						
						//Now that we have all the other required data, just figure out the node connections
						JArray cN_JSON = JArray.Parse(nodeJSON["cN"].ToString());
						foreach(JObject tempNode in cN_JSON) {
							nC_List.Add(
								new NodeConnection(
									g.nodes[(int)tempNode["ID"]],
									(int)tempNode["W"]
								)
							);
						}
						
						n.cN = new NodeConnection[nC_List.Count()];
						for(int i = 0; i < nC_List.Count(); i++) { n.cN[i] = nC_List[i]; }
					}
					
					g.nodes[ID] = n;
					if (n.nII) { for (int i = 0; i < g.inputNodes.Length;  i++) { g.inputNodes[i]  = n; } }
					if (n.nIO) { for (int i = 0; i < g.outputNodes.Length; i++) { g.outputNodes[i] = n; } }
				}
			} else {
				JObject rawGenomeData = (JObject)JsonConvert.DeserializeObject(File.ReadAllText(dir));
				g.nodes = new Node[rawGenomeData.Count];
				
				Log.Error("CryptoAI","Could not complete loading network in dumpAndPump mode. W.I.P");
				quit();
			}
			
			return g;
		}
		
		public static Network LoadNetwork_MT(string dir) { SetDirectory(dir); return LoadNetwork_MT(); }
		public static Network LoadNetwork_MT() {
			if (saveDirectory == "") {
				Console.ForegroundColor = ConsoleColor.Red;
				Log.Error("CryptoAI","Could not load network. LoadNetwork was called without setting save directory");
				quit();
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
						saveFileRepair();
						Log.Print("AI Config","Loading AI network from backup save");
						loadDirectory = netBackupDir;
					} else if (result == 2) {
						Log.Warning("AI Config","Backup save corrupted. Attempting file repair");
						saveFileRepair();
						Log.Print("AI Config","Loading AI network from primary save");
						loadDirectory = netSaveDir;
					} else {
						Log.Error("AI Config","Invalid save status detected in save directory. Defaulting to primary save");
						quit();
					}
				} catch {
					//null
					Log.Warning("AI Config","Error reading from save status file. Defaulting to primary save");
					loadDirectory = netSaveDir;
					File.WriteAllText(netStatusFile,"0");
					Log.Success("AI Config","Overwrote corrupt data in save file");
				}
			} else {
				Log.Error("CryptoAI","Could not find save status file. Using default option");
				if (Directory.Exists(netSaveDir)) {
					Log.Print("AI Config","Loading AI network from primary save");
					loadDirectory = netSaveDir;
				} else if (Directory.Exists(netBackupDir)) {
					Log.Print("AI Config","Loading AI network from backup save");
					loadDirectory = netBackupDir;
				} else {
					Log.Error("CryptoAI","Could not load network. No save folder detected in directory");
					quit();
				}
			}
			
			string[] allGenomeDirectories = null;
			Network N = new Network();
			if (dumpAndPump) {
				allGenomeDirectories = Directory.GetFiles(loadDirectory,"*.json");
			} else {
				allGenomeDirectories = Directory.GetDirectories(loadDirectory,"*",SearchOption.TopDirectoryOnly);
			}
			
			N.genomes = new Genome[allGenomeDirectories.Length];
			
			Stopwatch stopwatch = new Stopwatch();
			stopwatch.Start();
			
			List<ThreadDataContainer3> TDC_List = new List<ThreadDataContainer3>();
			bool run = true;
			int genomeIndex = 0;
			int maxThreadCount = 32;
			int currentThreadCount = 0;
			while(run) {
				while (currentThreadCount < maxThreadCount) {
					if(genomeIndex == N.genomes.Length) { run = false; break; }
				
					//HERE WE WANT TO ACTUALLY START THE THREAD PROCESS
					ThreadClass p = new ThreadClass();
					ThreadDataContainer3 TDC = new ThreadDataContainer3();
					
					TDC.g = new Genome();
					TDC.dir = allGenomeDirectories[genomeIndex];
					TDC.ID = genomeIndex;
					TDC_List.Add(TDC);
					Thread workerThread = new Thread(p.LoadGenome_CPUMultithreaded);  
					workerThread.Start(TDC);
						
					currentThreadCount++;
					genomeIndex++;
				}
				
				if (run) {
					for (int x = 0; x < TDC_List.Count; x++) {
						if (TDC_List[x].threadCompletionStatus) {
							N.genomes[TDC_List[x].ID] = TDC_List[x].g;
							TDC_List.RemoveAt(x);
							currentThreadCount--;
							x--;
							if (once2) {
								once2 = false;
								double predictedTime = stopwatch.Elapsed.Milliseconds * (double)N.genomes.Length/(double)maxThreadCount;
								//PrintFormattedMsg("AI Config","DEBUG","Predicted time for this process: " + predictedTime);
								stopwatch.Stop();
							}
						}
					}
				}
			}
			while(TDC_List.Count > 0) {
				for (int x = 0; x < TDC_List.Count; x++) {
					if (TDC_List[x].threadCompletionStatus) {
						N.genomes[TDC_List[x].ID] = TDC_List[x].g;
						TDC_List.RemoveAt(x);
						currentThreadCount--;
						x--;
						if (once2) {
							once2 = false;
							double predictedTime = stopwatch.Elapsed.Milliseconds * (double)N.genomes.Length/(double)maxThreadCount;
							//PrintFormattedMsg("AI Config","DEBUG","Predicted time for this process: " + predictedTime);
							stopwatch.Stop();
						}
					}
				}
			}
			
			Log.Success("CryptoAI","Loaded AI network");
			return N;
		}
		
		public static void CopyFilesRecursively(string sourcePath, string targetPath) {
			//Now Create all of the directories
			foreach (string dirPath in Directory.GetDirectories(sourcePath, "*", SearchOption.AllDirectories)) {
				Directory.CreateDirectory(dirPath.Replace(sourcePath, targetPath));
			}

			//Copy all the files & Replaces any files with the same name
			foreach (string newPath in Directory.GetFiles(sourcePath, "*.*",SearchOption.AllDirectories)) {
				File.Copy(newPath, newPath.Replace(sourcePath, targetPath), true);
			}
		}
		
		public static void saveFileRepair() {
			if (saveDirectory == "") {
				Console.ForegroundColor = ConsoleColor.Red;
				Log.Error("File Repair","Could not repair save files. saveFileRepair was called without setting save directory");
				quit();
			}
			
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
					} else if (result == 1) {
						//Primary save is corrupted
						Log.Error("File Repair","Detected primary save file corruption");
						Log.Print("File Repair","Attempting to fix save files");
						
						if (Directory.Exists(netSaveDir))
							Directory.Delete(netSaveDir,true);
						Directory.CreateDirectory(netSaveDir);
						Log.Print("File Repair","Copying from backup");
						CopyFilesRecursively(netBackupDir,netSaveDir);
						Log.Print("File Repair","Resetting file save status");
						File.WriteAllText(netStatusFile,"0");
						Log.Success("File Repair","Save files repaired!");
					} else if (result == 2) {
						//Backup save is corrupted
						Log.Error("File Repair","Detected backup save file corruption");
						Log.Print("File Repair","Attempting to fix save files");
						
						if (Directory.Exists(netBackupDir))
							Directory.Delete(netBackupDir,true);
						Directory.CreateDirectory(netBackupDir);
						Log.Print("File Repair","Copying from primary");
						CopyFilesRecursively(netSaveDir,netBackupDir);
						Log.Print("File Repair","Resetting file save status");
						File.WriteAllText(netStatusFile,"0");
						Log.Success("File Repair","Save files repaired!");
					} else {
						Log.Error("AI Config","Invalid save status detected in save directory. Defaulting to primary save");
						quit();
					}
				} catch {
					//Could not read from status file
					Log.Error("File Repair","An internal error has occured while repairing save data");
					quit();
				}
			} else {
				Log.Error("AI Config","Could not find save status file");
				quit();
			}
		}
		
		public static bool clearSave() { return clearSave(true); }
		public static bool clearSave(bool askUser) {
			if (saveDirectory == "") {
				Log.Error("CryptoAI","Could not remove save file as save directory has not been set!");
				quit();
			}
			
			if (!Directory.Exists(saveDirectory)) {
				Log.Error("CryptoAI","Could not remove save file as no save was found at directory: " + saveDirectory);
				return true;
			}
			
			//Delete the save directory from the disk. Completely wipe it
			if (askUser) {
				Log.Error("CryptoAI","Are you absolutely certain you want to clear the saved AI network? (Y/N)");
				char userInput = Console.ReadKey(true).KeyChar;
				while (userInput != 'y' && userInput != 'n') { userInput = Console.ReadKey(true).KeyChar; }
				if (userInput == 'n') {
					Log.Print("CryptoAI","User cancelled save data deletion");
					return false;
				}
			}
			
			Log.Print("CryptoAI","Deleting AI Network save data from disk permanently");
			Directory.Delete(saveDirectory,true);
			Log.Success("CryptoAI","Deleted AI save data");
			Directory.CreateDirectory(saveDirectory);
			return true;
		}
		
		public static void SaveNetwork(Network N) {
			//We cannot save the AI if we don't have a directory to save it to
			if (saveDirectory == "") {
				Log.Error("CryptoAI","Could not save network. SaveNetwork was called without setting save directory");
				quit();
			}
			
			string netSaveDir = saveDirectory + "Network Save Orig/";
			string netBackupDir = saveDirectory + "Network Save Backup/";
			string netStatusFile = saveDirectory + "saveStatus.txt";
			
			//double sizeOfSingleGenome = JsonConvert.SerializeObject(N.genomes[0]).Length;
			//double genomeCount = N.inputNodesCount + N.outputNodesCount + N.hiddenLayerCount*N.hiddenLayerWidth;
			//double sizeOnDisk = genomeCount*2*sizeOfSingleGenome/1000000000;
			//Log.Debug("CryptoAI","Saving AI network. This process using the JSON serializer would be estimated to take up: " + sizeOnDisk.ToString() + " GB of disk space");
			if (!Directory.Exists(saveDirectory))
				Directory.CreateDirectory(saveDirectory);
			
			File.WriteAllText(saveDirectory + "Network Info.json","[\n\t\"Iterations\":0\n]");
			
			//Save the original copy
			File.WriteAllText(netStatusFile,"1");
			if (Directory.Exists(netSaveDir)) {
				Log.Print("CryptoAI","Clearing old primary save");
				Directory.Delete(netSaveDir,true);
				Log.Success("CryptoAI","Cleared old primary save");
			}
			
			Log.Print("CryptoAI","Saving AI network - primary copy");
			Directory.CreateDirectory(netSaveDir);
			SaveNetworkInternal_CPU_MT(N,netSaveDir);
			Log.Success("CryptoAI","Saved AI network - primary copy");
			
			//Save the backup copy
			
			File.WriteAllText(netStatusFile,"2");
			if (Directory.Exists(netBackupDir)) {
				Log.Print("CryptoAI","Clearing old backup save");
				Directory.Delete(netBackupDir,true);
				Log.Success("CryptoAI","Cleared old backup save");
			}
			Log.Print("CryptoAI","Saving AI network - backup copy");
			Directory.CreateDirectory(netBackupDir);
			SaveNetworkInternal_CPU_MT(N,netBackupDir);
			Log.Success("CryptoAI","Saved AI network - backup copy");
			
			//Now finish up
			File.WriteAllText(netStatusFile,"0");
			Log.Success("CryptoAI","Saved AI network!");
		}
		
		public static void SaveNetworkInternal_CPU_MT(Network N, string dir) {
			//Here, all of the other stuff regarding backup copies and user interfaces is done and dealt with
			//All that's left is the raw save object code. Go nuts!
			if (N == null) {
				Log.Error("CryptoAI","Could not save network. Null AI network cannot be saved");
				quit();
			}
			
			//Get the best performing genome's ID
			int bestGenomeID = 0;
			for(int i = 0; i < N.genomes.Length; i++) {
				if (N.genomes[i].fitness > N.genomes[bestGenomeID].fitness)
					bestGenomeID = N.genomes[i].ID;
			}
			File.WriteAllText(dir + "Best Genome.txt",bestGenomeID.ToString());
			
			
			Stopwatch stopwatch = new Stopwatch();
			stopwatch.Start();
			
			List<ThreadDataContainer2> TDC_List = new List<ThreadDataContainer2>();
			bool run = true;
			int genomeIndex = 0;
			int maxThreadCount = 8;
			int currentThreadCount = 0;
			while(run) {
				while (currentThreadCount < maxThreadCount) {
					if(genomeIndex == N.genomes.Length) { run = false; break; }
				
					//HERE WE WANT TO ACTUALLY START THE THREAD PROCESS
					ThreadClass p = new ThreadClass();
					ThreadDataContainer2 TDC = new ThreadDataContainer2();
					
					TDC.g = N.genomes[genomeIndex];
					TDC.ID = genomeIndex;
					TDC.dir = dir;
					TDC.GenomeCount = N.genomes.Length;
					TDC_List.Add(TDC);
					Thread workerThread = new Thread(p.SaveGenome_CPUMultithreaded);  
					workerThread.Start(TDC);
						
					currentThreadCount++;
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
		
		public static void quit() { exit(); }
		public static void exit() {
			Console.ForegroundColor = ConsoleColor.Magenta;
			Console.WriteLine();
			Console.WriteLine();
			Console.WriteLine();
			Console.WriteLine("--------------------------------------");
			Console.WriteLine("Please press Enter to quit the program");
			Console.ForegroundColor = ConsoleColor.White;
			Console.ReadLine();
			System.Environment.Exit(1);
		}
		[System.Obsolete("Use the Log class instead.")]
		public static void PrintFormattedMsg(string moduleName, string msgType, string msg) => Log.PrintFormattedMsg(moduleName, msgType, msg);
		
		public static Network NewNetwork_CPU_MT(int genomeCount, int inputNodes, int nodesPerHiddenLayer, int hiddenLayerCount, int outputNodes) {
			Log.Warning("AI Config","Generating new AI Network. This may take awhile...");
			
			Network N = new Network();
			N.inputNodesCount = inputNodes;
			N.outputNodesCount = outputNodes;
			N.hiddenLayerCount = hiddenLayerCount;
			N.hiddenLayerWidth = nodesPerHiddenLayer;
			
			N.genomes = new Genome[genomeCount];
			
			string[] genomeActivationTypes = new string[] { "classic","neural" };
			AI_Internal_Core.getRandomFloat();
			
			List<ThreadDataContainer> TDC_List = new List<ThreadDataContainer>();
			bool run = true;
			int genomeIndex = 0;
			int maxThreadCount = 8;
			int currentThreadCount = 0;
			while(run) {
				while (currentThreadCount < maxThreadCount) {
					if(genomeIndex == genomeCount) { run = false; break; }
				
					//HERE WE WANT TO ACTUALLY START THE THREAD PROCESS
					ThreadClass p = new ThreadClass();
					ThreadDataContainer TDC = new ThreadDataContainer();
					TDC.CPU_Thread_ID = genomeIndex;
					TDC.inputNodes = inputNodes;
					TDC.outputNodes = outputNodes;
					TDC.nodesPerHiddenLayer = nodesPerHiddenLayer;
					TDC.hiddenLayerCount = hiddenLayerCount;
					
					int randomIndex = AI_Internal_Core.rand.Next(genomeActivationTypes.Length);
					TDC.NodeActivationType = genomeActivationTypes[randomIndex];
					TDC_List.Add(TDC);
					Thread workerThread = new Thread(p.CreateNewRandomGenome_CPUMultithreaded);  
					workerThread.Start(TDC);
						
					currentThreadCount++;
					genomeIndex++;
				}
				
				if (run) {
					for (int x = 0; x < TDC_List.Count; x++) {
						if (TDC_List[x].threadCompletionStatus) {
							N.genomes[TDC_List[x].CPU_Thread_ID] = TDC_List[x].threadOutput;
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
						N.genomes[TDC_List[x].CPU_Thread_ID] = TDC_List[x].threadOutput;
						TDC_List.RemoveAt(x);
						currentThreadCount--;
						x--;
					}
				}
			}
			
			Log.Success("AI Config","Created new AI network");
			int[] genomeTypeCounts = new int[genomeActivationTypes.Length];
			for (int x = 0; x < N.genomes.Length; x++) {
				for(int i = 0; i < genomeTypeCounts.Length; i++) {
					if (N.genomes[x].nodes[0].NodeActivationType == genomeActivationTypes[i]) {
						N.genomes[x].GenomeType = N.genomes[x].nodes[0].NodeActivationType;
						genomeTypeCounts[i] += 1;
					}
				}
			}
			
			for(int i = 0; i < genomeTypeCounts.Length; i++) {
				Log.Print("AI Config","Genome Type: " + genomeActivationTypes[i] + "\t\tQuantity: " + genomeTypeCounts[i]);
			}
			
			return N;
		}
		
		public static int CheckNetworkIntegrity_CPU(Network N) {
			int nullDetected = 0;
			for(int i = 0; i < N.genomes.Length; i++) {
				if (N.genomes[i] == null) {
					Log.Error("AI Config","Genome " + (i+1).ToString() + "/" + N.genomes.Length.ToString() + ":\t\tStatus - Null");
					nullDetected++;
				} else {
					Log.Success("AI Config","Genome " + (i+1).ToString() + "/" + N.genomes.Length.ToString() + ":\t\tStatus - Functional\t\tID: " + N.genomes[i].ID);
				}
			}
			return nullDetected;
		}
		
		public static void CalculateNextNetworkIteration_CPU_MT(ref Network N) {
			Log.Print("CryptoAI","Iterating network");
			
			List<ThreadDataContainer3> TDC_List = new List<ThreadDataContainer3>();
			bool run = true;
			int genomeIndex = 0;
			int maxThreadCount = 8;
			int currentThreadCount = 0;
			while(run) {
				while (currentThreadCount < maxThreadCount) {
					if(genomeIndex == N.genomes.Length) { run = false; break; }
				
					//HERE WE WANT TO ACTUALLY START THE THREAD PROCESS
					ThreadClass p = new ThreadClass();
					ThreadDataContainer3 TDC = new ThreadDataContainer3();
					
					TDC.g = N.genomes[genomeIndex];
					TDC.ID = genomeIndex;
					TDC_List.Add(TDC);
					Thread workerThread = new Thread(p.IterateNetwork_CPUMultithreaded);  
					workerThread.Start(TDC);
						
					currentThreadCount++;
					genomeIndex++;
				}
				
				if (run) {
					for (int x = 0; x < TDC_List.Count; x++) {
						if (TDC_List[x].threadCompletionStatus) {
							N.genomes[TDC_List[x].g.ID] = TDC_List[x].g;
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
						N.genomes[TDC_List[x].g.ID] = TDC_List[x].g;
						TDC_List.RemoveAt(x);
						currentThreadCount--;
						x--;
					}
				}
			}
		}
		
		public static void RegenerateAllGenomes_CPU(ref Network N) {
			Log.Debug("CryptoAI","Regenerating genomes...");
			for (int gI = 0; gI < N.genomes.Length; gI++) {
				Genome g = N.genomes[gI];
				for (int nI = 0; nI < g.nodes.Length; nI++) {
					if (!g.nodes[nI].nII) {
						for (int i = 0; i < g.nodes[nI].cN.Length; i++) {
							g.nodes[nI].cN[i].w = AI_Internal_Core.getRandomFloat();
						}
					}
				}
			}
		}
	}
	
	public class ThreadClass {
		public void CreateNewRandomGenome_CPUMultithreaded(object data) {
			ThreadDataContainer TDC = (ThreadDataContainer)data;
			
			Genome g = new Genome();
			g.ID = TDC.CPU_Thread_ID;
			g.nodes = new Node[TDC.inputNodes + TDC.outputNodes + TDC.nodesPerHiddenLayer*TDC.hiddenLayerCount];
			g.inputNodes = new Node[TDC.inputNodes];
			g.outputNodes = new Node[TDC.outputNodes];
			g.GenomeType = TDC.NodeActivationType;
			
			//Create Input Nodes
			Node[] previousLayer = new Node[TDC.inputNodes];
			for (int i = 0; i < TDC.inputNodes; i++) {
				Node n = new Node(i,null);
				n.nII = true;
				n.NodeActivationType = TDC.NodeActivationType;
				g.inputNodes[i] = n;
				g.nodes[i] = n;
				previousLayer[i] = n;
			}
			
			//Create Hidden Layers
			int startingIndex = TDC.inputNodes;
			for(int layer = 0; layer < TDC.hiddenLayerCount; layer++) {
				for(int i = 0; i < TDC.nodesPerHiddenLayer; i++) {
					NodeConnection[] nc = new NodeConnection[previousLayer.Length];
					for (int connection = 0; connection < nc.Length; connection++) {
						nc[connection] = new NodeConnection(previousLayer[connection],AI_Internal_Core.getRandomFloat());
					}
					
					Node n = new Node(startingIndex + i + layer*TDC.nodesPerHiddenLayer,nc);
					n.NodeActivationType = TDC.NodeActivationType;
					n.tT = AI_Internal_Core.getRandomFloat();
					n.tS = AI_Internal_Core.getRandomFloat();
					g.nodes[startingIndex + i + layer*TDC.nodesPerHiddenLayer] = n;	//------------------------------------------------------------
				}
			}
			
			//Create output layers
			startingIndex += TDC.nodesPerHiddenLayer*TDC.hiddenLayerCount;
			for (int i = 0; i < TDC.outputNodes; i++) {
				NodeConnection[] nc = new NodeConnection[previousLayer.Length];
				for (int connection = 0; connection < nc.Length; connection++) {
					nc[connection] = new NodeConnection(previousLayer[connection],AI_Internal_Core.getRandomFloat());
				}
					
				Node n = new Node(startingIndex + i,nc);
				n.NodeActivationType = TDC.NodeActivationType;
				n.tT = AI_Internal_Core.getRandomFloat();
				n.tS = AI_Internal_Core.getRandomFloat();
				n.nIO = true;
				g.nodes[i + startingIndex] = n;
				g.outputNodes[i] = n;
			}
			
			TDC.threadOutput = g;
			TDC.threadCompletionStatus = true;
		}
		
		public void SaveGenome_CPUMultithreaded(object data) {
			ThreadDataContainer2 TDC = (ThreadDataContainer2)data;
			
			string genomeDirectory = TDC.dir;
			
			double bytesTakenByGenome = 0;
			
			if (NetworkInterface.dumpAndPump) {
				//Console.WriteLine(genomeDirectory + "Genome " + TDC.g.ID + ".json");
				
				string nodeFileData = "[\n\t";
				for(int i = 0; i < TDC.g.nodes.Length; i++) {
					Node n = TDC.g.nodes[i];
					nodeFileData += "{\n";
					nodeFileData += "\t\t\"ID\":" + n.ID.ToString() + ",\n";
					nodeFileData += "\t\t\"tT\":" + n.tT.ToString() + ",\n";
					nodeFileData += "\t\t\"tS\":" + n.tS.ToString() + ",\n";
					nodeFileData += "\t\t\"nII\":\"" + n.nII.ToString() + "\",\n";
					nodeFileData += "\t\t\"nIO\":\"" + n.nIO.ToString() + "\",\n";
					nodeFileData += "\t\t\"nIV\":" + n.nIV.ToString() + ",\n";
					nodeFileData += "\t\t\"nAT\":\"" + n.NodeActivationType + "\",\n";
					nodeFileData += "\t\t\"pO\":" + n.pO.ToString();
					
					if (n.cN == null) {
						//There are no connected nodes
						nodeFileData += "\n\t}";
					} else if (n.cN.Length == 0) {
						//There are no connected nodes
						nodeFileData += "\n\t}";
					} else {
						//There are connected nodes
						nodeFileData += ",\n";
						nodeFileData += "\t\t\"cN\":[\n";
						
						for(int x = 0; x < n.cN.Length; x++) {
							nodeFileData += "\t\t\t{\"ID\":" + n.cN[x].n.ID.ToString() + ",\"W\":" + n.cN[x].w + "}";
							if (x + 1 != n.cN.Length)
								nodeFileData += ",";
							nodeFileData += "\n";
						}
						
						nodeFileData += "\t\t]";
						nodeFileData += "\n\t}";
					}
					
					if (i + 1 != TDC.g.nodes.Length)
						nodeFileData += ",";
				}
				nodeFileData += "]";
				
				string fileDir = genomeDirectory + "Genome " + TDC.g.ID + ".json";
				bool tmpRun = true;
				while(tmpRun) {
					try {
						File.WriteAllText(fileDir,nodeFileData);
						tmpRun = false;
					} catch {
						Console.WriteLine("Could not access directory: " + fileDir);
						Thread.Sleep(5000);
					}
				}
			} else {
				//Console.WriteLine(genomeDirectory + "Genome " + TDC.g.ID + "/");
				Directory.CreateDirectory(genomeDirectory + "Genome " + TDC.g.ID + "/");
				string fileDir = genomeDirectory + "Genome " + TDC.g.ID + "/Genome Activation Type.txt";
				bool tmpRun = true;
				while(tmpRun) {
					try {
						File.WriteAllText(fileDir,TDC.g.GenomeType);
						tmpRun = false;
					} catch {
						Console.WriteLine("Could not access directory: " + fileDir);
						Thread.Sleep(5000);
					}
				}
				for(int i = 0; i < TDC.g.nodes.Length; i++) {
					Node n = TDC.g.nodes[i];
					string nodeFileData = "{\n";
					nodeFileData += "\t\"ID\":" + n.ID.ToString() + ",\n";
					nodeFileData += "\t\"tT\":" + n.tT.ToString() + ",\n";
					nodeFileData += "\t\"tS\":" + n.tS.ToString() + ",\n";
					nodeFileData += "\t\"nII\":\"" + n.nII.ToString() + "\",\n";
					nodeFileData += "\t\"nIO\":\"" + n.nIO.ToString() + "\",\n";
					nodeFileData += "\t\"nIV\":" + n.nIV.ToString() + ",\n";
					nodeFileData += "\t\"nAT\":\"" + n.NodeActivationType + "\",\n";
					nodeFileData += "\t\"pO\":" + n.pO.ToString();
					if (n.cN == null) {
						//There are no connected nodes
						nodeFileData += "\n}";
					} else if (n.cN.Length == 0) {
						//There are no connected nodes
						nodeFileData += "\n}";
					} else {
						//There are connected nodes
						nodeFileData += ",\n";
						nodeFileData += "\t\"cN\":[\n";
						
						for(int x = 0; x < n.cN.Length; x++) {
							nodeFileData += "\t\t{\"ID\":" + n.cN[x].n.ID.ToString() + ",\"W\":" + n.cN[x].w + "}";
							if (x + 1 != n.cN.Length)
								nodeFileData += ",";
							nodeFileData += "\n";
						}
						
						nodeFileData += "\t]\n}";
					}
					
					fileDir = genomeDirectory + "Genome " + TDC.g.ID + "/Node " + n.ID.ToString() + ".json";
					tmpRun = true;
					while(tmpRun) {
						try {
							File.WriteAllText(fileDir,nodeFileData);
							tmpRun = false;
						} catch {
							Console.WriteLine("Could not access directory: " + fileDir);
							Thread.Sleep(5000);
						}
					}
					
					bytesTakenByGenome += nodeFileData.Length;
				}
				
				//if (NetworkInterface.once) {
				//	NetworkInterface.once = false;
				//	bytesTakenByGenome/=10000000;
				//	NetworkInterface.PrintFormattedMsg("AI Config","DEBUG","Total AI network predicted size on disk: " + bytesTakenByGenome.ToString() + " GB");
				//}
			}
			
			TDC.threadCompletionStatus = true;
		}
		
		public void LoadGenome_CPUMultithreaded(object data) {
			ThreadDataContainer3 TDC = (ThreadDataContainer3)data;
			Genome g = TDC.g;
			
			if (!NetworkInterface.dumpAndPump) {
				g.GenomeType = File.ReadAllText(TDC.dir + "/Genome Activation Type.txt").Replace("\t","").Replace("\n","").Replace(" ","");
				g.ID = TDC.ID;
				
				string[] allNodeDirs = Directory.GetFiles(TDC.dir,"Node*.json");
				g.nodes = new Node[allNodeDirs.Length];
				for(int i = 0; i < g.nodes.Length; i++) {
					g.nodes[i] = new Node(i,null);
				}
				
				//Just sort the node directories in ascending order
				for (int i = 0; i < allNodeDirs.Length; i++) {
					allNodeDirs[i] = TDC.dir + "/Node " + i.ToString() + ".json";
				}
				
				foreach (string nodeDir in allNodeDirs) {
					JObject nodeJSON = JObject.Parse(File.ReadAllText(nodeDir));
					int ID = (int)nodeJSON["ID"];
					
					Node n = new Node(ID,new NodeConnection[0]);
					//Does this node already exist in g.nodes? If so, overwrite "n"
					for(int i = 0; i < g.nodes.Length; i++) {
						if (g.nodes[i].ID == ID)
							n = g.nodes[i];
					}
					
					n.NodeActivationType = g.GenomeType;
					//n.NodeActivationType = (string)nodeJSON["nAT"];
					
					n.tT = (double)nodeJSON["tT"];
					n.tS = (double)nodeJSON["tS"];
					n.nII = (bool)nodeJSON["nII"];
					n.nIO = (bool)nodeJSON["nIO"];
					n.nIV = (double)nodeJSON["nIV"];
					n.pO = (double)nodeJSON["pO"];
					
					if (!n.nII) {
						List<NodeConnection> nC_List = new List<NodeConnection>();
						
						//Now that we have all the other required data, just figure out the node connections
						JArray cN_JSON = JArray.Parse(nodeJSON["cN"].ToString());
						foreach(JObject tempNode in cN_JSON) {
							nC_List.Add(
								new NodeConnection(
									g.nodes[(int)tempNode["ID"]],
									(int)tempNode["W"]
								)
							);
						}
						
						n.cN = new NodeConnection[nC_List.Count()];
						for(int i = 0; i < nC_List.Count(); i++) { n.cN[i] = nC_List[i]; }
					}
					
					g.nodes[ID] = n;
					if (n.nII) { for (int i = 0; i < g.inputNodes.Length;  i++) { g.inputNodes[i]  = n; } }
					if (n.nIO) { for (int i = 0; i < g.outputNodes.Length; i++) { g.outputNodes[i] = n; } }
				}
			} else {
				JObject rawGenomeData = (JObject)JsonConvert.DeserializeObject(File.ReadAllText(TDC.dir));
				g.nodes = new Node[rawGenomeData.Count];
				
				Log.Error("CryptoAI","Could not complete loading network in dumpAndPump mode. W.I.P");
				NetworkInterface.quit();
			}
			
			TDC.threadCompletionStatus = true;
		}
		
		public void IterateNetwork_CPUMultithreaded(object data) {
			ThreadDataContainer3 TDC = (ThreadDataContainer3)data;
			
			if (TDC.g.fitness < TDC.g.prev_fitness)
				TDC.g.resetGenomeToPreviousState();
				
			for (int nI = 0; nI < TDC.g.nodes.Length; nI++) {
				TDC.g.nodes[nI].prev_tT = TDC.g.nodes[nI].tT;
				TDC.g.nodes[nI].prev_tS = TDC.g.nodes[nI].tS;
				TDC.g.nodes[nI].tT = Math.Max(Math.Min(TDC.g.nodes[nI].tT + 0.0001f * (AI_Internal_Core.getRandomFloat() - 0.5f),1),0);
				TDC.g.nodes[nI].tS = Math.Max(Math.Min(TDC.g.nodes[nI].tS + 0.0001f * (AI_Internal_Core.getRandomFloat() - 0.5f),1),0);
				
				if (!TDC.g.nodes[nI].nII) {
					for (int i = 0; i < TDC.g.nodes[nI].cN.Length; i++) {
						TDC.g.nodes[nI].cN[i].prev_w = TDC.g.nodes[nI].cN[i].w;
						TDC.g.nodes[nI].cN[i].w = Math.Max(Math.Min(TDC.g.nodes[nI].cN[i].w + 0.0001f * (AI_Internal_Core.getRandomFloat() - 0.5f),1),0);
					}
				}
			}
			
			TDC.threadCompletionStatus = true;
		}
	}
		
	public class ThreadDataContainer {
		public bool threadCompletionStatus = false;
		public int CPU_Thread_ID;
		public int inputNodes;
		public int outputNodes;
		public int nodesPerHiddenLayer;
		public int hiddenLayerCount;
		public string NodeActivationType;
		public Genome threadOutput;
	}
	
	public class ThreadDataContainer2 {
		public bool threadCompletionStatus = false;
		public Genome g;
		public int ID;
		public string dir;
		public int GenomeCount;
	}
	
	public class ThreadDataContainer3 {
		public bool threadCompletionStatus = false;
		public Genome g;
		public string dir;
		public int ID;
	}
}



















