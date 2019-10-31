using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PBRParaSet : MonoBehaviour
{
    [SerializeField,Range(0,1)]
    public float _Roughness;
    [SerializeField, Range(0, 1)]
    public float _Metallic;

    public Renderer render;

    private MaterialPropertyBlock materialProperty;

    private void Start()
    {
        materialProperty = new MaterialPropertyBlock();
    }
    private void Update()
    {
        materialProperty.SetFloat("_Roughness", _Roughness);
        materialProperty.SetFloat("_Metallic", _Metallic);
        render.SetPropertyBlock(materialProperty);
    }
}

