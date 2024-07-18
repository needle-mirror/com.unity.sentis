#if UNITY_EDITOR && UNITY_2023_2_OR_NEWER && ENABLE_CLOUD_SERVICES_ANALYTICS
using UnityEditor;
using System;
using UnityEngine.Analytics;

namespace Unity.Sentis
{
    // More information on how it interacts with the server can be found here: https://docs.editor-data.unity3d.com/Contribute/EditorAnalytics/cs_guide/
    // How to debug: https://docs.editor-data.unity3d.com/Contribute/EditorAnalytics/debugger_guide/
    // Make sure to update the version if you duplicate the schema instead of updating it.
    [AnalyticInfo(eventName: k_EventName, vendorKey: k_VendorKey, version: 3)]
    internal class SentisAnalytics : IAnalytic
    {
        private const string k_EventName = "sentisModelImport";
        private const string k_VendorKey = "unity.sentis";

        // This class is used to store the data that will be sent to the server. It must match the data that the server expects. Make sure you can read the data in the BigQuery.
        // More information on how to update the data: https://docs.dp.unity3d.com/Schema_Management/schemata_ui/
        [Serializable]
        internal class Data : IAnalytic.IData
        {
            public string[] allOperators;
            public int[] importWarningSeverity;
            public string[] importWarningMessages;
            public int modelLayerCount;
        }

        internal Data m_Data;

        public bool TryGatherData(out IAnalytic.IData data, out Exception error)
        {
            error = null;
            data = m_Data;
            return data != null;
        }

        public static void SendEvent(Data data)
        {
            SentisAnalytics analytic = new SentisAnalytics();
            analytic.m_Data = data;
            EditorAnalytics.SendAnalytic(analytic);
        }
    }
}
#endif
