using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Unity.Sentis.Compiler.Analyser
{
    static class GraphLogicAnalysis
    {
        public static int GetDownStreamLayersCount(Model model, string index)
        {
            return model.layers.Count(x => x.inputs.Contains(index));
        }
    }
}

