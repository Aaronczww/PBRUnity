// Upgrade NOTE: replaced '_World2Object' with 'unity_WorldToObject'

Shader "Unlit/PBR"
{
    Properties
    {
        _MainTex ("BaseColor", 2D) = "white"{}
        _RoughTex("Roughness", 2D) = "white"{}
        _MetallicTex("_Metallic", 2D) = "white"{}
        _Occulision("Ao",2D) = "white"{}
        _NormalTex("_NormalMap",2D) = "white"{}
        _CubeMap("environment map",Cube) = ""{}
        _Cubemap2("environment map2",Cube) = ""{} 
        _LuTMap("_LuTMap",2D) = "white"{}
        _Smoothness("_Smoothness",float) = 0.05
        metallic("_Metallic",float) = 0.05
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
            #include "Lighting.cginc"
            #include "UnityStandardBRDF.cginc" 
            #define PI 3.1415

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
                float4 tangent : TANGENT;
                float3 normal:NORMAL;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 position : SV_POSITION;
                float3 worldPos : TEXCOORD1;
                float3 worldNormal:NORMAL;
                float4 localPos : TEXCOORD3;

                float3 tspace0:TEXCOORD4;
                float3 tspace1:TEXCOORD5;
                float3 tspace2:TEXCOORD6;
                float3 localNormal:TEXCOORD7;
            };

            sampler2D _MainTex;
            sampler2D _RoughTex;
            sampler2D _MetallicTex;
            sampler2D _Occulision;
            sampler2D _NormalTex;
            sampler2D _LuTMap;
            samplerCUBE _CubeMap;
            samplerCUBE _Cubemap2;

            float4 _LuTMap_ST;
            float4 _MainTex_ST;
            float4 _NormalTex_ST;
            float4 _RoughTex_ST;
            float4 _MetallicTex_ST;
            float4 _Occulision_ST;

            float _Smoothness;
            float metallic;
            
            float3 diffuseColor(float3 diffuseColor,float Roughness,float NoV,float NoL,float VoH)
            {
                float Fd = 0.5 + 2 * Roughness * VoH * VoH;
                float Fl = 1 + (Fd-1)*(1-pow(NoL,5));
                float Fv = 1 + (Fd-1)*(1-pow(NoV,5));
                float3 diffuse = (diffuseColor / PI) * Fl * Fv;
                return diffuse;
            }

            //F项

            float3 fresnel(float VoH,float F0)
            {
                float3 F = F0 + (1 - F0) * exp2((-5.55473 * VoH - 6.98316) * VoH);
                return F;
            }

            // D项
            float D_GTR2(float Roughness,float NoH)
            {
                float a2 = Roughness*Roughness;
                float cos2th = NoH * NoH;
                float den = (1.0 + (a2 - 1.0) * cos2th);

                return a2  / (PI * den * den);
            }

            float D_GGX_TR(float3 N,float3 H,float Roughness)
            {
                float a = Roughness * Roughness;
                float NdotH =  max(dot(N,H),0);
                float b = NdotH * NdotH;
                float denom = (b * (a-1)+1);

                return a/(3.14*denom*denom);
            }

            //G项
            float GeometrySchlickGGX(float NdotV,float k)
            {
                float nom = NdotV;
                float denom = NdotV * (1-k) + k;
                return nom/denom;
            }

            float GeometrySmith(float3 N,float3 V,float3 L,float k)
            {
                    float NdotV = max(dot(N, V), 0.0);
                    float NdotL = max(dot(N, L), 0.0);
                    float ggx1 = GeometrySchlickGGX(NdotV,k);
                    float ggx2 = GeometrySchlickGGX(NdotL,k);
                    return ggx1*ggx2;
            }

            float3 diffuseIrradiance(samplerCUBE cubeMap,float3 worldNormal)
            {
                float3 diffuseIrradiance = float3(0,0,0);
                //环境贴图卷积
                float sampleDelta = 0.025;
                float nSamples = 0;

                float3 n = normalize(worldNormal);
                float3 up    = float3(0.0, 1.0, 0.0);
                float3 right = cross(up, n);
                up           = cross(n, right);

                for(float phi=0;phi<2*PI;phi+=sampleDelta)
                {
                    for(float theata=0;theata<0.5*PI;theata+=sampleDelta)
                    {
                        //球面坐标到笛卡尔坐标
                        float3 tangentSample = float3(sin(theata)*cos(phi),sin(theata)*sin(phi),cos(theata));
                        
                        // tangentSample.x = dot(Tspace0,tangentSample.x);
                        // tangentSample.y = dot(Tspace1,tangentSample.y);
                        // tangentSample.z = dot(Tspace2,tangentSample.z);

                        float3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * n; 
                        
                        diffuseIrradiance = diffuseIrradiance + texCUBE(cubeMap,normalize(sampleVec)).xyz * cos(theata) * sin(theata);
                        nSamples++;
                    }
                }
                return PI*diffuseIrradiance*(1/nSamples);
            }

            //low-discrepancy sequence

            //Radical Inversion 
            float IntegerRadicalInverse(int Base,int i)
            {
                int numPoints,inverse;
                numPoints = 1;
                for(inverse = 0;i>0;i/=Base)
                {
                    inverse = inverse * Base + (i % Base);
                    numPoints = numPoints * Base;
                }
                return inverse/(float)numPoints;
            } 

            float VanDerCorpus(int n,int base)
            {
                float invBase = 1.0/(float)base;
                float denom = 1.0;
                float result = 0.0;
                for(int i=0;i<32;++i)
                {
                    if(n>0)
                    {
                        denom = fmod(float(n),2.0);
                        result += denom * invBase;
                        invBase = invBase / 2.0;
                        n = int(float(n)/2.0);
                    }
                }
                return result;
            }

            //Hammersley点集
            float2 Hammerseley(int i,int N)
            {
                return float2(float(i)/float(N),VanDerCorpus(i,2)); 
            }

            //重要性采样返回半程向量
            float3 ImportanceSampleGGX(float2 Xi,float3 N,float roughness)
            {
                float a = roughness * roughness;

                float phi = 2.0 * PI * Xi.x;
                float cosTheta = sqrt((1-Xi.y)/(1.0+(a*a-1)*Xi.y));
                float sinTheta = sqrt(1.0-cosTheta*cosTheta);

                float3 H;
                H.x = cos(phi) * sinTheta;
                H.y = sin(phi) * sinTheta;
                H.z = cosTheta;

                float3 up = abs(N.z) < 0.999 ? float3(0,0,1):float3(1,0,0);
                float3 tangent = normalize(cross(up,N));
                float3 bitangent = cross(N,tangent);

                float3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
                return normalize(sampleVec);
            }

            //prefilteredColor
            float3 prefilteredColor(float3 WorldPos,samplerCUBE cubeMap,float roughness)
            {
                float3 N = normalize(WorldPos);
                float3 R = N;
                float3 V = R;

                int SampleCount = 1024;
                float totalWeight = 0.0;
                float3 result = float3(0,0,0);
                float k = roughness * roughness / 2.0;

                for(int i=0;i<SampleCount;i++)
                {
                    float2 Xi = Hammerseley(i,SampleCount);
                    float3 H = ImportanceSampleGGX(Xi,N,roughness);
                    float3 L = normalize(2.0*dot(V,H)*H - V);

                    float NdotL = max(dot(N,L),0);
                    if(NdotL > 0.0)
                    {
                        float D = GeometrySmith(N,V,L,k);
                        float NdotH = max(dot(N,H),0);
                        float HdotV = max(dot(H,V),0);

                        float pdf = D * NdotH / ((4*HdotV) + 0.0001);
                        float saSample = 1.0/(float(SampleCount) * pdf + 0.0001);
                        float mipLevel = roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / (2/1980*1080));

                        result = result + texCUBE(cubeMap,L).xyz * NdotL;
                        //result = result + texCUBElod(_CubeMap,L,mipLevel).xyz * NdotL;
                        totalWeight = totalWeight + NdotL;
                    }
                }

                return (result / totalWeight);
            }

            //BRDF LUT Map
            float3 IntegrateBRDF(float NdotV,float roughness)
            {
                float3 V;
                V.x = sqrt(1-NdotV*NdotV);
                V.y = 0.0;
                V.z = NdotV;

                float A = 0.0;
                float B = 0.0;
                float C = 0.0;

                float3 N = float3(0,0,1.0);
                int SampleCount = 1024;
                float3 result = float3(0,0,0);
                
                for(int i=0;i<SampleCount;++i)
                {
                    float2 Xi = Hammerseley(i,SampleCount);
                    float3 H = ImportanceSampleGGX(Xi,N,roughness);
                    float3 L = normalize(2.0*dot(V,H)*H - V);

                    float NdotL = max(L.z,0);
                    float NdotH = max(H.z,0);
                    float VdotH = max(dot(V,H),0);

                    if(NdotL > 0.0)
                    {
                        float G = GeometrySmith(N,V,L,roughness);
                        float G_Vis = (G * VdotH) / (NdotH * NdotV);
                        float Fc = pow(1-VdotH,5);

                        A = A + (1-Fc) * G_Vis;
                        B = B + Fc * G_Vis;
                    }
                }
                result = float3 (A,B,0);
                return result /(float)SampleCount;
            }

            v2f vert (appdata v)
            {
                v2f o;
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.position = UnityObjectToClipPos(v.vertex);

                o.worldPos = mul(v.vertex, unity_WorldToObject).xyz;
                o.worldNormal = normalize(mul(v.normal,(float3x3)unity_WorldToObject));
                o.localPos = v.vertex;
                
                //切线转到世界空间
                float3 wTangent = mul(v.tangent,unity_WorldToObject).xyz;
                
                half tangentSign = v.tangent.w * unity_WorldTransformParams.w;
                half3 wBitangent = cross(o.worldNormal, wTangent) * tangentSign;

                o.tspace0 = float3(wTangent.x,wBitangent.x,o.worldNormal.x);
                o.tspace1 = float3(wTangent.y,wBitangent.y,o.worldNormal.y);
                o.tspace2 = float3(wTangent.z,wBitangent.z,o.worldNormal.z);
                o.localNormal = v.normal;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {                

                //float4 cubeMap = texCUBE(_CubeMap,normalize(i.localPos.xyz));
                samplerCUBE cube;

                float4 baseColor = tex2D(_MainTex, i.uv);
                float4 rougness = tex2D(_RoughTex,i.uv);
                float4 metallicColor = tex2D(_MetallicTex,i.uv);
                float4 aoColor = tex2D(_Occulision,i.uv);

                float ao = aoColor.x;

                float3 lightDir = normalize(_WorldSpaceLightPos0.xyz - i.worldPos.xyz);
                float3 viewDir = normalize(_WorldSpaceCameraPos.xyz - i.worldPos.xyz);
                float3 normalDir = i.worldNormal;
                float3 halfDir = normalize(lightDir + viewDir);
                float3 ReflectDir = reflect(-lightDir,normalDir);
                
                float NoV = max(dot(normalDir,viewDir),0);
                float NoL = max(dot(normalDir,lightDir),0);
                float VoH = max(dot(halfDir,viewDir),0);
                float NoH = max(dot(normalDir,halfDir),0);

                float wn = max(dot(lightDir,normalDir),0);
                float wi = max(dot(viewDir,normalDir),0);

                float3 F0 = lerp(unity_ColorSpaceDielectricSpec.rgb, baseColor, metallic);
	            
                float perceptualRoughness = 1 - _Smoothness;
	            float a = perceptualRoughness * perceptualRoughness;

                float3 F = fresnel(NoV,F0);
                //IBL
                float3 EnvDiffuse = texCUBE(_CubeMap,normalDir);

                float3 MprefilteredColor = texCUBE(_CubeMap,float3(ReflectDir));

                float3 temp = texCUBE(_Cubemap2,float3(ReflectDir));

                MprefilteredColor = lerp(MprefilteredColor,temp,a);

                float3 scale_bias = tex2Dbias(_LuTMap,float4(NoV,NoV,a,a)).xyz;
                float3 Envspecluar = MprefilteredColor * (F0 * scale_bias.x + scale_bias.y);

                float D = D_GGX_TR(normalDir,halfDir,a);
                D = D_GTR2(a,NoH);

                //直接光重映射
                float k = (a+1)*(a+1)/8;
                //IBL光重映射
                //k = (a * a)/2;
                float G = GeometrySmith(normalDir,viewDir,lightDir,k);

                float3 DFG = D*F*G;

                float  denominator =  4 * wi * wn + 0.001;
                float3 spec = DFG/denominator;

                float3 ks = F;
                float3 kd = float3(1.0,1.0,1.0) - ks;
                kd =  kd * (1-metallic);

                float3 ambient = kd * EnvDiffuse + Envspecluar;

                float3 diffuse = diffuseColor(baseColor,a,NoV,NoL,VoH);

                float distance = length(lightDir);
                float attenuation = 1.0 / (distance * distance);
                float3 radians = float3(1,1,1);

                float3 Lo = (diffuse + spec) * radians * max(dot(normalDir,lightDir),0);

                float3 color = float3(Lo + ambient);

                //映射到HDR
                color = color / (color + float3(1.0,1.0,1.0));
                color = pow(color, float(1.0/2.2));

                float4 col = float4(color,1.0) * float4(1,0,0,0);
                return col;
            }
            ENDCG
        }
    }
}
