//https://rapidapi.com/blog/how-to-use-an-api-with-c-sharp/

using System;
using System.IO;
using System.Net;
using System.Threading;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;

using unirest_net.http;

using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using IronXL;

using CryptoAI;

//Kline/Candlestick data 			GET		/api/v3/klines
//New order 						POST 	/api/v3/order 		(HMAC SHA256)
//Test new order 					POST 	/api/v3/order/test 	(HMAC SHA256)
//Query order						GET		/api/v3/order		(HMAC SHA256)
//Cancel order						DELETE	/api/v3/order		(HMAC SHA256)
//Cancel all orders					DELETE  /api/v3/openOrders  (HMAC SHA256)
//Currently open orders				GET 	/api/v3/openOrders  (HMAC SHA256)
//All orders						GET 	/api/v3/allOrders 	(HMAC SHA256)
//etc                               etc		etc					etc

namespace Binance_API {
    public static class API {
		public static string filePath = "Z:/SaveData/";
		static string key = "wWsD2wSqNNyyzAdk6RNeHwrF0chaREPLTIQBKAKUe6QkNlXXS4eEK5zEn5IKu0I7";
		static string secret = "IHMgTvIRQgcV0I3UiqFZqoj647K4YsaTC9DsEPK6v01HSYbsgb9h3NEiU9HEssiR";
		public static bool createdJSON = false;
		public static bool realTrading = false;
		public static List<double[]> allCoinsData = null;
		public static JObject ExchangeInfo;
		
		public static bool getServerConnectivity() {
			try {
				//HttpResponse<string> response = Unirest.get("https://api.binance.com/api/v3/time").asJson<string>();
				Unirest.get("https://api.binance.com/api/v3/time");
			} catch { return false;}
			return true;
		}
		
		public static double getServerTime() {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			HttpResponse<string> response;
			while(true) {
				try {
					response = Unirest.get("https://api.binance.com/api/v3/time").asJson<string>();
					break;
				} catch {}
			}
			
			return float.Parse(JObject.Parse(response.Body.ToString())["time"].ToString());
		}
		
		public static string getExchangeInfo() {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			while(true) {
				try {
					string result = Unirest.get("https://api.binance.com/api/v3/exchangeInfo").asJson<string>().Body.ToString();
					ExchangeInfo = (JObject)JsonConvert.DeserializeObject(result);
					return result;
				} catch {}
			}
		}
		
		public static string getOrderBook(string symbol) { return getOrderBook(symbol,-1); }
		public static string getOrderBook(string symbol, int limit) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/depth?symbol=" + symbol;
			if (limit > 0) { resultingString += "&limit=" + limit.ToString(); }
			
			string result;
			while(true) {
				try {
					result = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch {}
			}
			
			return result;
		}
		
		public static string getRecentTrades(string symbol) { return getRecentTrades(symbol,-1); }
		public static string getRecentTrades(string symbol, int limit) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/trades?symbol=" + symbol;
			if (limit > 0) { resultingString += "&limit=" + limit.ToString(); }
			
			string response = "";
			while(true) {
				try {
					response = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch { Console.WriteLine("[API Interface][Warning]: Invalid response from server. Trying again..."); }
			}
			
			return response;
		}
		
		public static string getHistoricTrades() { return getHistoricTrades("LTCBTC",500,-1); }
		public static string getHistoricTrades(string symbol) { return getHistoricTrades(symbol,500,-1); }
		public static string getHistoricTrades(string symbol, int limit) { return getHistoricTrades(symbol,limit,-1); }
		public static string getHistoricTrades(string symbol, int limit, long fromId) {
			if (!createdJSON) {
				createdJSON = true;
				if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
				
				string resultingString = "https://api.binance.com/api/v3/historicalTrades?symbol=" + symbol;
				if (limit > 0) { resultingString += "&limit=" + limit.ToString(); }
				if (fromId > 0) { resultingString += "&fromId=" + fromId.ToString(); }
				
				string fileDataResult2;
				while(true) {
					try {
						fileDataResult2 = Unirest.get(resultingString).header("X-MBX-APIKEY",key).asJson<string>().Body.ToString();
						break;
					} catch {}
				}
				
				try {
					File.WriteAllText(filePath + "HistoricDataCache.json", fileDataResult2);
				} catch {
					Log.Error("API Interface", "Failed to save historic data cache to directory: " + filePath + "HistoricDataCache.json");
					Log.Error("API Interface", "Please check that the file path is accessable, and this program has permissions to write to it");
					NetworkInterface.quit();
				}
			}
			
			string fileDataResult = "";
			StreamReader file = new StreamReader(filePath + "HistoricDataCache.json");
			string line = file.ReadLine();
			while (line != null) {
				fileDataResult += line + "\n";
				line = file.ReadLine();
			}
				
			//Now we just empty out the last of it and close the file
			file.Close();
			
			return fileDataResult;
		}
		
		public static string getAggregateTrades(string symbol)                         { return getAggregateTrades(symbol,0,0,0,0); }
		public static string getAggregateTrades(string symbol, int limit)              { return getAggregateTrades(symbol,0,limit,0,0); }
		public static string getAggregateTrades(string symbol, long fromID)            { return getAggregateTrades(symbol,fromID,0,0,0); }
		public static string getAggregateTrades(string symbol, long fromID, int limit) { return getAggregateTrades(symbol,fromID,limit,0,0); }
		public static string getAggregateTrades(string symbol, long fromID, int limit, long startTime, long endTime) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/aggTrades?symbol=" + symbol;
			if (fromID > 0)    { resultingString += "&fromID="    + fromID.ToString();    }
			if (startTime > 0) { resultingString += "&startTime=" + startTime.ToString(); }
			if (endTime > 0)   { resultingString += "&endTime="   + endTime.ToString();   }
			if (limit > 0)     { resultingString += "&limit="     + limit.ToString();     }
			
			string result;
			while(true) {
				try {
					result = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch {}
			}
			
			return result;
		}
		
		public static string getCurrentAveragePrice(string symbol) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/avgPrice?symbol=" + symbol;
			
			string result;
			while(true) {
				try {
					result = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch {}
			}
			
			return result;
		}
		
		//With this function, we can get all supported symbols/currencies (Extended data)
		public static string get24hrTickerPriceChange(string symbol) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/ticker/24hr";
			if (symbol != "") { resultingString += "?symbol=" + symbol; }
			
			string result;
			while(true) {
				try {
					result = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch {}
			}
			
			return result;
		}
		
		//With this function, we can get all supported symbols/currencies (Compact data)
		public static string getTickerPrice(string symbol) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/ticker/price";
			if (symbol != "") { resultingString += "?symbol=" + symbol; }
			
			string result;
			while(true) {
				try {
					result = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch {}
			}
			
			return result;
		}
		
		public static string getSymbolOrderBook(string symbol) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/ticker/bookTicker";
			if (symbol != "") { resultingString += "?symbol=" + symbol; }
			
			string result;
			while(true) {
				try {
					result = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch {}
			}
			
			return result;
		}
		
		public static string makeSecureCall(string URL) { return makeSecureCall(URL,"","GET"); }
		public static string makeSecureCall(string URL, string totalParams) { return makeSecureCall(URL,totalParams,"GET"); }
		public static string makeSecureCall(string URL, string totalParams, string side) {
			while (true) {
				string resultingURL = string.Empty;
				try {
					if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
					
					totalParams += "&timestamp=" + getServerTime();
					
					HMACSHA256 myhmacsha256 = new HMACSHA256(Encoding.UTF8.GetBytes(secret));
					byte[] encodedData = Encoding.UTF8.GetBytes(totalParams);
					byte[] hash = myhmacsha256.ComputeHash(encodedData);
					
					string signature = string.Empty;
					foreach	(byte x in hash) { signature += String.Format("{0:x2}", x); }
					
					if (totalParams != "") {
						resultingURL = URL + "?" + totalParams + "&signature=" + signature;
					} else {
						resultingURL = URL + "?signature=" + signature;
					}
					
					string result;
					if (side == "GET") {
						while(true) {
							try {
								result = Unirest.get(resultingURL).header("X-MBX-APIKEY",key).asJson<string>().Body.ToString();
								break;
							} catch {}
						}
						
						return result;
					}
					
					while(true) {
						try {
							result = Unirest.post(resultingURL).header("X-MBX-APIKEY",key).asJson<string>().Body.ToString();
							break;
						} catch {}
					}
					
					return result;
				} catch {
					Thread.Sleep(2000);
				}
			}
		}
		
		public static string getKlines(string symbol, string interval) { return getKlines(symbol,interval,0); }
		public static string getKlines(string symbol, string interval, int limit) {
			if (!getServerConnectivity()) { throw new Exception("No Server Connection"); }
			
			string resultingString = "https://api.binance.com/api/v3/klines?symbol=" + symbol + "&interval=" + interval;
			if (limit != 0) { resultingString += "&limit=" + limit.ToString(); }
			
			string result;
			while(true) {
				try {
					result = Unirest.get(resultingString).asJson<string>().Body.ToString();
					break;
				} catch {}
			}
			
			return result;
		}
		
		//public enum Sides {
		//	BUY,
		//	SELL
		//};
		//public enum Types {
		//	LIMIT,
		//	MARKET,
		//	STOP_LOSS,
		//	STOP_LOSS_LIMIT,
		//	TAKE_PROFIT,
		//	TAKE_PROFIT_LIMIT,
		//	LIMITMAKER
		//};
		public static string createNewBuySellOrder(string symbol, string side, float quantity) {
			if (realTrading) {
				string tmp = API.makeSecureCall(
					"https://api.binance.com/api/v3/order"
					, "symbol=" + symbol
					+ "&side=" + side
					+ "&type=MARKET"
					+ "&quoteOrderQty=" + quantity
					//+ "&quantity=" + quantity
					, "POST"
				);
				SaveTradeToJsonDocument(symbol, side, quantity);
				return tmp;
			} else {
				SaveTradeToJsonDocument(symbol, side, quantity);
				return null;
			}
		}
		
		public static void SaveTradeToJsonDocument(string symbol, string side, float quantity) {
			if (!Directory.Exists(filePath + "Crypto Trades/"))
				Directory.CreateDirectory(filePath + "Crypto Trades/");
			
			int tradeID = 0;
			string directory = filePath + "Crypto Trades/Crypto Trade" + tradeID + ".json";
			while (File.Exists(directory)) {
				tradeID++;
				directory = filePath + "Crypto Trades/Crypto Trade" + tradeID + ".json";
			}
			
			File.Create(directory);
			File.WriteAllText(directory,
							  "{\n\t\"Date UTC\":" + DateTime.UtcNow.ToLongDateString() + 
							  ",\n\t\"Time UTC\":" + DateTime.UtcNow.TimeOfDay.ToString() +
							  ",\n\t\"Date Local\":" + DateTime.Now.ToLongDateString() +
							  ",\n\t\"Time Local\":" + DateTime.Now.TimeOfDay.ToString() +
							  ",\n\t\"Symbol\":" + symbol +
							  ",\n\t\"BuyOrSell\":\"" + side +
							  "\",\n\t\"Quantity\"" + quantity.ToString() +
							  "\n}"
			);
		}
		
		public static float getWalletContents(string symbol) {
			string API_result = API.makeSecureCall("https://api.binance.com/api/v3/allOrders", "symbol=" + symbol);
			//Console.WriteLine(API_result);
			//saveDebugFile(API_result,"C:\\Users\\aj200\\Desktop\\Backups\\2. C# CNN\\walletInfo.debug");
			
			if (API_result.Length < 10)
				return 0;
			
			try {
				List<JObject> outputResult = JsonConvert.DeserializeObject<List<JObject>>(API_result);
				float currentWalletValue = 0;
				foreach(JObject i in outputResult) {
					currentWalletValue += float.Parse(JObject.Parse(i.ToString())["price"].ToString());
				}
				return currentWalletValue;
			} catch { return -1; }
		}
		
		public static void saveDebugFile(string data, string filePath) {
			File.WriteAllText(filePath, "");
			using (StreamWriter sw = File.AppendText(filePath)) {
				foreach(string x in data.Split(",")) {
					sw.WriteLine("\t" + x + ",");
				}
			}
		}
		
		public static void UpdateAllCoinsData() { UpdateAllCoinsData(1); }
		public static void UpdateAllCoinsData(int minimumDataCount) {
			allCoinsData = new List<double[]>();
			getExchangeInfo();
			JArray jArr = (JArray)ExchangeInfo["symbols"];
			for (int i = 0; i < jArr.Count; i++) {
				string symbol = jArr[i]["symbol"].ToString();
				string symbolData = getHistoricTrades(symbol,10000);
				JArray tmp = (JArray)JsonConvert.DeserializeObject(symbolData);
				
				if (tmp.Count >= minimumDataCount) {
					double[] result = new double[tmp.Count];
					for(int x = 0; x < tmp.Count; x++) { result[x] = (double)tmp[x]["price"]; }
				
					allCoinsData.Add(result);
				}
			}
		}
    }
}