using System;
using System.Collections.Generic;

namespace CryptoAI {

    /// <summary>
    /// Used for logging to the console. This could be adapted later for writing to files and filtering logs.
    /// </summary>
    public class Log {
        public static bool PrintTime = true;
        public static bool PrintModule = false;

        public enum MsgType {
            Log,
            Debug,
            Success,
            Warning,
            Error,
        }

        // For performance and simplicity pre-computed strings for msg type.
        static string[] typeText = new string[] {
            "  Log  ",
            " Debug ",
            "Success",
            "Warning",
            " Error ",
        };
        static ConsoleColor[] colours = new ConsoleColor[] {
            ConsoleColor.White,
            ConsoleColor.Cyan,
            ConsoleColor.Green,
            ConsoleColor.Yellow,
            ConsoleColor.Red
        };

        /// <summary>
        /// Write a message to console.
        /// </summary>
        /// <param name="source">Where the message came from.</param>
        /// <param name="level">The kind/Level of message.</param>
        /// <param name="msg">The message to send.</param>
        public static void Write(string source, MsgType level, string msg) {
            //Example:  [01:02:27][ ERROR ] Hey all, Scott here. This is bad, real bad.

            if(PrintTime)
                Console.Write($"{DateTime.Now:HH:mm:ss} ");

            if (PrintModule)
                Console.Write($"[{source}] ");

            Console.Write('[');
            Console.ForegroundColor = colours[(int)level];
            Console.Write(typeText[(int)level]);
            Console.ResetColor();
            Console.Write("] ");

            Console.WriteLine(msg);
        }

        /// <summary>
        /// Write a message to console.
        /// </summary>
        /// <param name="source">The type/class that the message came from.</param>
        /// <param name="level">The Kind/Level of message.</param>
        /// <param name="msg">The message to send.</param>
        public static void Write(Type source, MsgType level, string msg)
            => Write(source.Name, level, msg);

        //public static void Write(object obj, MsgType type, string msg)
        //    => Write(obj.ToString(), type, msg);

        // Shorthands for write.
        public static void Print(string source, string msg)
            => Write(source, MsgType.Log, msg);
        public static void Debug(string source, string msg)
            => Write(source, MsgType.Debug, msg);
        public static void Success(string source, string msg)
            => Write(source, MsgType.Success, msg);
        public static void Warning(string source, string msg)
            => Write(source, MsgType.Warning, msg);
        public static void Error(string source, string msg)
            => Write(source, MsgType.Error, msg);
        
        // Exceptions.
        public static void Exception(string source, Exception e)
            => Write(source, MsgType.Error, e.ToString());

        // The old way of doing it for legacy code and habits.
        [System.Obsolete("Use Write() instead.")]
        public static void PrintFormattedMsg(string moduleName, string msgType, string msg) {
            var type = MsgType.Log;
            Enum.TryParse<MsgType>(msgType, true, out type);
            Write(moduleName, type, msg);
        }
    }
}