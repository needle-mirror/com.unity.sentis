using System;
using System.Collections.Generic;
using Unity.Sentis.Layers;
using System.Reflection;
using System.Collections;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class RemoveDuplicateLayersPass : IModelPass
    {
        long GetHashCode(Layer layer, Dictionary<int, int> duplicateConstants)
        {
            long seed = 0;
            HashHelper.HashCombine(ref seed, layer.GetType());
            foreach (var input in layer.inputs)
            {
                var remappedInput = duplicateConstants.GetValueOrDefault(input, input);
                HashHelper.HashCombine(ref seed, remappedInput);
            }

            return seed;
        }

        List<object> GetComparableFields(Layer layer)
        {
            var infos = new List<object>();
            var fields = layer.GetType().GetFields(BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            foreach (var field in fields)
            {
                var name = field.Name;
                if (name == "outputs" || name == "inputs")
                    continue;
                infos.Add(field.GetValue(layer));
            }
            return infos;
        }

        bool AllEqual(List<object> l0, List<object> l1)
        {
            if (l0.Count != l1.Count)
                return false;

            for (int i = 0; i < l0.Count; i++)
            {
                var f0 = l0[i];
                var f1 = l1[i];

                if ((f0 is IList a0) && (f1 is IList a1))
                {
                    if (a0.Count != a1.Count)
                        return false;

                    for (int j = 0; j < a0.Count; j++)
                    {
                        var e0 = a0[j]; var e1 = a1[j];
                        if (!Equals(e0, e1))
                            return false;
                    }
                }
                else if (!Equals(f0, f1))
                    return false;
            }

            return true;
        }

        List<object> GetComparableFields(ref Dictionary<int, List<object>> comparableFieldsByLayer, Layer layer)
        {
            if (!comparableFieldsByLayer.TryGetValue(layer.outputs[0], out var layerFields))
            {
                layerFields = GetComparableFields(layer);
                comparableFieldsByLayer.Add(layer.outputs[0], layerFields);
            }

            return layerFields;
        }

        public void Run(ref Model model)
        {
            var duplicateConstants = DuplicateConstantCalculator.CalculateDuplicateConstants(model);

            // Algorithm: remove same layers
            // a layer is the same if it has the same types and all fields and inputs are the same
            // foreach layer:
            //  compute soft hash on layer inputs + type
            //  foreach collision:
            //    remove layer if equal (full param check) to collision
            var remapRemovedIndexes = new Dictionary<int, int>();
            var layersToRemove = new HashSet<int>();
            var layerByInput = new Dictionary<long, List<Layer>>();
            var comparableFieldsByLayer = new Dictionary<int, List<object>>();
            foreach (var layer in model.layers)
            {
                // in place input rename, to propagate removal stat mid traversal
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (remapRemovedIndexes.ContainsKey(input))
                        layer.inputs[i] = remapRemovedIndexes[input];
                }

                long hash = GetHashCode(layer, duplicateConstants);
                if (!layerByInput.TryGetValue(hash, out var collisionLayers))
                {
                    layerByInput.Add(hash, new List<Layer>() { layer });
                    continue;
                }

                var layerFields = GetComparableFields(ref comparableFieldsByLayer, layer);
                bool removed = false;
                foreach (var similarLayer in collisionLayers)
                {
                    List<object> fields = GetComparableFields(ref comparableFieldsByLayer, similarLayer);

                    if (!AllEqual(layerFields, fields))
                        continue;

                    if (layer is RandomLayer { hasSeed: false })
                        continue;

                    var inputsAllEqual = true;
                    for (int i = 0; i < layer.inputs.Length && inputsAllEqual; i++)
                        inputsAllEqual &= duplicateConstants.GetValueOrDefault(layer.inputs[i], layer.inputs[i]) == duplicateConstants.GetValueOrDefault(similarLayer.inputs[i], similarLayer.inputs[i]);
                    if (!inputsAllEqual)
                        continue;

                    remapRemovedIndexes.Add(layer.outputs[0], similarLayer.outputs[0]);

                    layersToRemove.Add(layer.outputs[0]);
                    removed = true;

                    if (layer.outputs.Length != similarLayer.outputs.Length)
                        break;

                    for (int i = 0; i < layer.outputs.Length; i++)
                    {
                        if (!remapRemovedIndexes.ContainsKey(layer.outputs[i]))
                            remapRemovedIndexes.Add(layer.outputs[i], similarLayer.outputs[i]);
                    }

                    break;
                }

                if (!removed)
                    collisionLayers.Add(layer);
            }

            model.layers.RemoveAll(l => layersToRemove.Contains(l.outputs[0]));

            // all inputs have been remapped in place, no need to update layers

            for (var i = 0; i < model.outputs.Count; i++)
            {
                if (remapRemovedIndexes.TryGetValue(model.outputs[i].index, out var remappedIndex))
                {
                    model.outputs[i] = new Model.Output{
                        name = model.outputs[i].name,
                        index = remappedIndex
                    };
                }
            }
        }
    }

    static class DuplicateConstantCalculator
    {
        static long GetHashCode(Constant constant)
        {
            long seed = 0;
            HashHelper.HashCombine(ref seed, constant.shape);

            if (constant.shape.HasZeroDims())
                return seed;

            for (int i = 0; i < constant.shape.length; i++)
                HashHelper.HashCombine(ref seed, constant.weights.Get<int>(i));

            return seed;
        }

        static bool AreEqual(Constant c0, Constant c1)
        {
            if (c0.shape != c1.shape)
                return false;

            if (c0.shape.HasZeroDims() && c1.shape.HasZeroDims())
                return true;

            for (int i = 0; i < c0.shape.length; i++)
            {
                int v0 = c0.weights.Get<int>(i);
                int v1 = c1.weights.Get<int>(i);
                if (v0 != v1)
                    return false;
            }

            return true;
        }

        public static Dictionary<int, int> CalculateDuplicateConstants(Model model)
        {
            // Algorithm: remove same constant
            // a constant is the same if it's length/shape/weights are all identical
            // foreach constant:
            //  compute first soft hash on constant length
            //  if equal compute hash on constant weights
            //     check secondary hashmap on weight.hash
            //     if collision, hard comparison
            // N.B: no handling of potential wrong collision on weight.hash
            var constantsToRemove = new Dictionary<int, int>();
            var shapeHashTupleToConstants = new Dictionary<Tuple<TensorShape, long>, List<Constant>>();
            foreach (var constant in model.constants)
            {
                if (constant.dataType != DataType.Int)
                    continue;

                var key = new Tuple<TensorShape, long>(constant.shape, GetHashCode(constant));
                if (!shapeHashTupleToConstants.TryGetValue(key, out var potentialSimilarConstants))
                {
                    shapeHashTupleToConstants.Add(key, new List<Constant> { constant });
                    continue;
                }

                bool removed = false;
                foreach (var similarConstant in potentialSimilarConstants)
                {
                    // collision, double check values
                    if (!AreEqual(constant, similarConstant))
                        continue;

                    removed = true;
                    constantsToRemove.Add(constant.index, similarConstant.index);
                    break;
                }

                if (!removed)
                    potentialSimilarConstants.Add(constant);
            }

            return constantsToRemove;
        }
    }

    class RemoveDuplicatesPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var removeLayers = new RemoveDuplicateLayersPass();
            removeLayers.Run(ref model);
        }
    }
}
