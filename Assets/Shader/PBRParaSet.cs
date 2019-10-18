using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PBRParaSet : MonoBehaviour
{
    [SerializeField,Range(0,1)]
    public float _Smoothness;
    [SerializeField, Range(0, 1)]
    public float metallic;

    public Renderer render;

    private MaterialPropertyBlock materialProperty;

    private void Start()
    {
        materialProperty = new MaterialPropertyBlock();
    }
    private void Update()
    {
        materialProperty.SetFloat("_Smoothness", _Smoothness);
        materialProperty.SetFloat("metallic", metallic);
        render.SetPropertyBlock(materialProperty);
    }
}

