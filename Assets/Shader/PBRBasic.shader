// Upgrade NOTE: replaced '_World2Object' with 'unity_WorldToObject'

Shader "PBR/Basic"
{
    Properties
    {
        _NormalTex("_NormalMap",2D) = "white"{}
        _CubeMap("environment map",Cube) = ""{}
        _LuTMap("_LuTMap",2D) = "white"{}
        _Roughness("_Roughness",float) = 0.05
        _Metallic("_Metallic",float) = 0.05
        _Tint("_Tint Color",Color) = (1, 0, 0, 1)
        _F0("F0",Vector) = (0.5,0.5,0.5)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Lighting On
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "PBRCommon.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float3 normal:NORMAL;
                float4 tangent:TANGENT;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 position : SV_POSITION;
                float3 worldPos : TEXCOORD1;
                float3 worldNormal:TEXCOORD2;
                float4 world_tangent:TEXCOORD3;
            };

            sampler2D _NormalTex;
            sampler2D _LuTMap;
            samplerCUBE _CubeMap;
            float _Roughness;
            float _Metallic;
            float4 _Tint;
            float3 _F0;

            v2f vert (appdata v)
            {
                v2f o;
                o.uv = v.uv;
                o.position = UnityObjectToClipPos(v.vertex);

                o.worldPos = mul(unity_ObjectToWorld, v.vertex).xyz;
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.world_tangent = float4(UnityObjectToWorldDir(v.tangent),v.tangent.w);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {                
                float3 vertexN = normalize(i.worldNormal);
                float3 T = normalize(i.world_tangent);
                float3 B = cross(vertexN,T) * i.world_tangent.w;

                float3x3 worldToTagent = float3x3(T,B,vertexN);
                float3 tangentN = UnpackNormal(tex2D(_NormalTex,i.uv));
                float3 normalDir = mul(tangentN,worldToTagent);

                normalDir = normalize(i.worldNormal);

                float3 lightDir = -normalize(_WorldSpaceLightPos0.xyz - i.worldPos.xyz);
                float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.worldPos.xyz);
                float3 halfDir = normalize(lightDir + viewDir);
                float3 ReflectDir = reflect(-viewDir,normalDir);

                float4 EnvLight = IBLEnvLight(_LuTMap,_CubeMap,_Roughness,normalDir,viewDir,ReflectDir,_Metallic,_Tint);

                float4 DirectLight = PBRDirectLight(normalDir,viewDir,lightDir,_F0,_Roughness,_Metallic,_Tint);
                
                float3 color = DirectLight.xyz;

                color = color / (color + float3(1.0,1.0,1.0));
                color = pow(color, float(1.0/2.2));

                return float4(color + EnvLight.xyz,1);
            }
            ENDCG
        }
    }
}
