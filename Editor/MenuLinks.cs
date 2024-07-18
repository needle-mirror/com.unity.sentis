using UnityEngine;
using UnityEditor;
using System;
using System.Reflection;

class MenuLinks : EditorWindow
{
    [MenuItem("Sentis/Online Documentation", false, 2000)]
    static void OnlineDocs()
    {
        Application.OpenURL("https://docs.unity3d.com/Packages/com.unity.sentis@latest");
    }

    [MenuItem("Sentis/Discussion Community", false, 2001)]
    static void On()
    {
        Application.OpenURL("https://discussions.unity.com/c/ai-beta/sentis/10");
    }

    [MenuItem("Sentis/Submit your Project", false, 2002)]
    static void SubmitProject()
    {
        Application.OpenURL("https://create.unity.com/sentis-project-submission");
    }
}
