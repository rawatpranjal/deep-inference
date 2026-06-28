# General PSD-Λ vs RieszNet vs Oracle — flawless inference across 3 GLM families (M=50)
### Linear DGP (lambdas=cholesky,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 1.0000, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.9993 | -0.0007 | 0.0548 | 0.0535 | 0.98 | 98% |
| FLM[cholesky] | 0.9931 | -0.0069 | 0.0568 | 0.0591 | 1.04 | 96% |
| FLM[oracle] | 1.0090 | +0.0090 | 0.0603 | 0.0588 | 0.97 | 96% |
| RieszNet | 0.9958 | -0.0042 | 0.0553 | 0.0606 | 1.10 | 100% |
| Naive | 0.9981 | -0.0019 | 0.0740 | 0.0121 | 0.16 | 18% |

[LOGIT]  (workers=8)
  rep 1/50: oracle=0.140 FLM[cholesky]=0.094 FLM[oracle]=0.175 riesz=0.192 naive=0.155
  rep 2/50: oracle=0.119 FLM[cholesky]=0.100 FLM[oracle]=0.123 riesz=0.216 naive=0.112
  rep 3/50: oracle=0.175 FLM[cholesky]=0.235 FLM[oracle]=0.179 riesz=0.184 naive=0.191
  rep 4/50: oracle=0.124 FLM[cholesky]=0.196 FLM[oracle]=0.141 riesz=0.148 naive=0.127
  rep 5/50: oracle=0.128 FLM[cholesky]=0.186 FLM[oracle]=0.153 riesz=0.122 naive=0.136
  rep 6/50: oracle=0.189 FLM[cholesky]=0.228 FLM[oracle]=0.199 riesz=0.155 naive=0.194
  rep 7/50: oracle=0.163 FLM[cholesky]=0.078 FLM[oracle]=0.180 riesz=0.163 naive=0.155
  rep 8/50: oracle=0.158 FLM[cholesky]=0.182 FLM[oracle]=0.178 riesz=0.163 naive=0.156
  rep 9/50: oracle=0.155 FLM[cholesky]=0.091 FLM[oracle]=0.176 riesz=0.165 naive=0.160
  rep 10/50: oracle=0.138 FLM[cholesky]=0.081 FLM[oracle]=0.174 riesz=0.134 naive=0.167
  rep 11/50: oracle=0.154 FLM[cholesky]=0.096 FLM[oracle]=0.181 riesz=0.154 naive=0.154
  rep 12/50: oracle=0.154 FLM[cholesky]=0.196 FLM[oracle]=0.161 riesz=0.158 naive=0.189
  rep 13/50: oracle=0.116 FLM[cholesky]=0.227 FLM[oracle]=0.137 riesz=0.150 naive=0.114
  rep 14/50: oracle=0.146 FLM[cholesky]=0.139 FLM[oracle]=0.143 riesz=0.129 naive=0.118
  rep 15/50: oracle=0.157 FLM[cholesky]=0.107 FLM[oracle]=0.190 riesz=0.155 naive=0.166
  rep 16/50: oracle=0.185 FLM[cholesky]=0.226 FLM[oracle]=0.197 riesz=0.221 naive=0.208
  rep 17/50: oracle=0.149 FLM[cholesky]=0.183 FLM[oracle]=0.175 riesz=0.172 naive=0.168
  rep 18/50: oracle=0.157 FLM[cholesky]=0.117 FLM[oracle]=0.127 riesz=0.152 naive=0.163
  rep 19/50: oracle=0.169 FLM[cholesky]=0.178 FLM[oracle]=0.180 riesz=0.171 naive=0.168
  rep 20/50: oracle=0.147 FLM[cholesky]=0.157 FLM[oracle]=0.150 riesz=0.139 naive=0.150
  rep 21/50: oracle=0.155 FLM[cholesky]=0.189 FLM[oracle]=0.164 riesz=0.170 naive=0.172
  rep 22/50: oracle=0.135 FLM[cholesky]=0.129 FLM[oracle]=0.113 riesz=0.118 naive=0.115
  rep 23/50: oracle=0.151 FLM[cholesky]=0.230 FLM[oracle]=0.170 riesz=0.154 naive=0.170
  rep 24/50: oracle=0.174 FLM[cholesky]=0.165 FLM[oracle]=0.186 riesz=0.082 naive=0.192
  rep 25/50: oracle=0.159 FLM[cholesky]=0.138 FLM[oracle]=0.171 riesz=0.167 naive=0.164
  rep 26/50: oracle=0.156 FLM[cholesky]=0.155 FLM[oracle]=0.154 riesz=0.129 naive=0.121
  rep 27/50: oracle=0.137 FLM[cholesky]=0.148 FLM[oracle]=0.139 riesz=0.163 naive=0.112
  rep 28/50: oracle=0.174 FLM[cholesky]=0.179 FLM[oracle]=0.182 riesz=0.155 naive=0.191
  rep 29/50: oracle=0.120 FLM[cholesky]=0.057 FLM[oracle]=0.112 riesz=0.100 naive=0.129
  rep 30/50: oracle=0.159 FLM[cholesky]=0.204 FLM[oracle]=0.163 riesz=0.150 naive=0.176
  rep 31/50: oracle=0.182 FLM[cholesky]=0.230 FLM[oracle]=0.179 riesz=0.183 naive=0.220
  rep 32/50: oracle=0.137 FLM[cholesky]=0.147 FLM[oracle]=0.142 riesz=0.138 naive=0.119
  rep 33/50: oracle=0.199 FLM[cholesky]=0.271 FLM[oracle]=0.246 riesz=0.201 naive=0.219
  rep 34/50: oracle=0.192 FLM[cholesky]=0.191 FLM[oracle]=0.186 riesz=0.165 naive=0.182
  rep 35/50: oracle=0.143 FLM[cholesky]=0.090 FLM[oracle]=0.106 riesz=0.127 naive=0.132
  rep 36/50: oracle=0.134 FLM[cholesky]=0.133 FLM[oracle]=0.166 riesz=0.154 naive=0.142
  rep 37/50: oracle=0.135 FLM[cholesky]=0.147 FLM[oracle]=0.150 riesz=0.133 naive=0.155
  rep 38/50: oracle=0.200 FLM[cholesky]=0.169 FLM[oracle]=0.219 riesz=0.193 naive=0.178
  rep 39/50: oracle=0.159 FLM[cholesky]=0.060 FLM[oracle]=0.170 riesz=0.058 naive=0.160
  rep 40/50: oracle=0.140 FLM[cholesky]=0.113 FLM[oracle]=0.150 riesz=0.133 naive=0.133
  rep 41/50: oracle=0.173 FLM[cholesky]=0.159 FLM[oracle]=0.181 riesz=0.166 naive=0.202
  rep 42/50: oracle=0.164 FLM[cholesky]=0.107 FLM[oracle]=0.154 riesz=0.160 naive=0.166
  rep 43/50: oracle=0.133 FLM[cholesky]=0.112 FLM[oracle]=0.143 riesz=0.122 naive=0.145
  rep 44/50: oracle=0.145 FLM[cholesky]=0.158 FLM[oracle]=0.161 riesz=0.146 naive=0.169
  rep 45/50: oracle=0.103 FLM[cholesky]=-0.007 FLM[oracle]=0.117 riesz=0.104 naive=0.091
  rep 46/50: oracle=0.140 FLM[cholesky]=0.124 FLM[oracle]=0.160 riesz=0.154 naive=0.137
  rep 47/50: oracle=0.165 FLM[cholesky]=0.121 FLM[oracle]=0.175 riesz=0.146 naive=0.149
  rep 48/50: oracle=0.166 FLM[cholesky]=0.187 FLM[oracle]=0.191 riesz=0.172 naive=0.159
  rep 49/50: oracle=0.161 FLM[cholesky]=0.078 FLM[oracle]=0.157 riesz=0.168 naive=0.149
  rep 50/50: oracle=0.201 FLM[cholesky]=0.190 FLM[oracle]=0.214 riesz=0.193 naive=0.180

### Logit DGP (lambdas=cholesky,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 0.1481, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.1543 | +0.0062 | 0.0224 | 0.0220 | 0.98 | 92% |
| FLM[cholesky] | 0.1494 | +0.0013 | 0.0560 | 0.0563 | 1.01 | 98% |
| FLM[oracle] | 0.1648 | +0.0167 | 0.0278 | 0.0238 | 0.86 | 88% |
| RieszNet | 0.1530 | +0.0049 | 0.0307 | 0.0399 | 1.30 | 100% |
| Naive | 0.1576 | +0.0095 | 0.0293 | 0.0030 | 0.10 | 12% |
wrote exploration/results.md
### Poisson DGP (lambdas=cholesky,oracle, n=2000, folds=10, tikhonov=0.01)  (truth = 0.8981, M = 50)

| method | mean est | bias | emp SE | mean est SE | SE ratio | coverage |
|---|---|---|---|---|---|---|
| Oracle | 0.8888 | -0.0093 | 0.0703 | 0.0682 | 0.97 | 100% |
| FLM[cholesky] | 0.8718 | -0.0263 | 0.1049 | 0.1106 | 1.05 | 98% |
| FLM[oracle] | 0.8555 | -0.0427 | 0.0843 | 0.0792 | 0.94 | 90% |
| RieszNet | 0.8755 | -0.0226 | 0.1086 | 0.1274 | 1.17 | 100% |
| Naive | 0.8758 | -0.0224 | 0.1025 | 0.0236 | 0.23 | 28% |
wrote exploration/results.md
