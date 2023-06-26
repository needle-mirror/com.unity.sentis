using System;
using UnityEngine.Assertions;

namespace Unity.Sentis {

static class Logger
{
    //TODO handle context (execution/import/model/layer) + log it along error/assert (warning think of thread safety vs context)
    //TODO is it valuable to have a way collect many errors before asserting/throw?
    public static void AssertAreEqual(object expected, object actual, string msg)
    {
        #if (UNITY_ASSERTIONS)
        if (expected != actual)
            Assert.AreEqual(expected, actual, msg);
        #endif
    }
    public static void AssertAreEqual(object expected, object actual, string msg, object msgParam)
    {
        #if (UNITY_ASSERTIONS)
        if (expected != actual)
            Assert.AreEqual(expected, actual, String.Format(msg, msgParam));
        #endif
    }
    public static void AssertAreEqual(object expected, object actual, string msg, object msgParam0, object msgParam1)
    {
        #if (UNITY_ASSERTIONS)
        if (expected != actual)
            Assert.AreEqual(expected, actual, String.Format(msg, msgParam0, msgParam1));
        #endif
    }
    public static void AssertAreEqual(object expected, object actual, string msg, object msgParam0, object msgParam1, object msgParam2)
    {
        #if (UNITY_ASSERTIONS)
        if (expected != actual)
            Assert.AreEqual(expected, actual, String.Format(msg, msgParam0, msgParam1, msgParam2));
        #endif
    }

    public static void AssertIsFalse(bool condition, string msg)
    {
        #if (UNITY_ASSERTIONS)
        if (!condition)
            Assert.IsFalse(condition, msg);
        #endif
    }

    public static void AssertIsTrue(bool condition, string msg)
    {
        #if (UNITY_ASSERTIONS)
        if (!condition)
            Assert.IsTrue(condition, msg);
        #endif
    }
    public static void AssertIsTrue(bool condition, string msg, object msgParam0)
    {
        #if (UNITY_ASSERTIONS)
        if (!condition)
            Assert.IsTrue(condition, String.Format(msg, msgParam0));
        #endif
    }
    public static void AssertIsTrue(bool condition, string msg, object msgParam0, object msgParam1)
    {
        #if (UNITY_ASSERTIONS)
        if (!condition)
            Assert.IsTrue(condition, String.Format(msg, msgParam0, msgParam1));
        #endif
    }
}
} // namespace Unity.Sentis

