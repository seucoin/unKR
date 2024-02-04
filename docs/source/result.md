# Results
There are some reproduced model results using unKR.

## CN15K

### Confidence prediction
<table>
    <thead>
    <tr>
        <th>Category</th>
        <th>Model</th>
        <th>MSE</th>
        <th>MAE</th>
    </tr>
    </thead>
    <tbody align="center" valign="center">
    <tr>
        <td rowspan="8">Normal model</td>
        <td>BEUrRE</td>
        <td>0.11656 </td>
        <td>0.28259 </td>
    </tr>
    <tr>
        <td>PASSLEAF_ComplEx</td>
        <td>0.23095 </td>
        <td>0.39970 </td>
    </tr>
    <tr>
        <td>PASSLEAF_DistMult</td>
        <td>0.21575 </td>
        <td>0.37923 </td>
    </tr>
    <tr>
        <td>PASSLEAF_RotatE</td>
        <td>0.09375 </td>
        <td>0.24801 </td>
    </tr>
    <tr>
        <td>UKGE</td>
        <td>0.24635 </td>
        <td>0.40864 </td>
    </tr>
    <tr>
        <td>UKGEPSL</td>
        <td>0.24630 </td>
        <td>0.40881 </td>
    </tr>
    <tr>
        <td>UKGsE</td>
        <td>0.10274 </td>
        <td>0.25564 </td>
    </tr>
    <tr>
        <td>UPGAT</td>
        <td>0.14933 </td>
        <td>0.30796 </td>
    </tr>
    <tr>
        <td>Few-shot model</td>
        <td>GMUC</td>
        <td>0.01700 </td>
        <td>0.10400</td>
    </tr>
    </tbody>
</table>

### Link prediction

#### All test data

##### Raw
<table>
    <thead>
    <tr>
        <th>Category</th>
        <th>Model</th>
        <th>Hits@1</th>
        <th>Hits@3</th>
        <th>Hits@10</th>
        <th>MRR</th>
        <th>MR</th>
        <th>WMRR</th>
        <th>WMR </th>
    </tr>
    </thead>
    <tbody align="center" valign="center">
    <tr>
        <td rowspan="10">Normal model</td>
        <td>BEUrRE</td>
        <td>0.030 </td>
        <td>0.127 </td>
        <td>0.248 </td>
        <td>0.103 </td>
        <td>1041.157 </td>
        <td>0.112 </td>
        <td>916.689  </td>
    </tr>
    <tr>
        <td>FocusE</td>
        <td>0.090 </td>
        <td>0.168 </td>
        <td>0.288 </td>
        <td>0.154 </td>
        <td>1760.524 </td>
        <td>0.168 </td>
        <td>1629.575  </td>
    </tr>
    <tr>
        <td>GTransE</td>
        <td>0.043 </td>
        <td>0.112 </td>
        <td>0.207 </td>
        <td>0.098 </td>
        <td>1114.496 </td>
        <td>0.105 </td>
        <td>978.835  </td>
    </tr>
    <tr>
        <td>PASSLEAF_ComplEx</td>
        <td>0.058 </td>
        <td>0.131 </td>
        <td>0.262 </td>
        <td>0.122 </td>
        <td>1175.950 </td>
        <td>0.136 </td>
        <td>1099.126  </td>
    </tr>
    <tr>
        <td>PASSLEAF_DistMult</td>
        <td>0.057 </td>
        <td>0.126 </td>
        <td>0.244 </td>
        <td>0.117 </td>
        <td>1203.598 </td>
        <td>0.131 </td>
        <td>1092.162  </td>
    </tr>
    <tr>
        <td>PASSLEAF_RotatE</td>
        <td>0.029 </td>
        <td>0.126 </td>
        <td>0.252 </td>
        <td>0.103 </td>
        <td>1020.154 </td>
        <td>0.112 </td>
        <td>832.611  </td>
    </tr>
    <tr>
        <td>UKGE</td>
        <td>0.056 </td>
        <td>0.116 </td>
        <td>0.209 </td>
        <td>0.106 </td>
        <td>1616.041 </td>
        <td>0.142 </td>
        <td>1467.336  </td>
    </tr>
    <tr>
        <td>UKGEPSL</td>
        <td>0.052 </td>
        <td>0.114 </td>
        <td>0.210 </td>
        <td>0.104 </td>
        <td>1581.759 </td>
        <td>0.116 </td>
        <td>1446.819  </td>
    </tr>
    <tr>
        <td>UKGsE</td>
        <td>0.002 </td>
        <td>0.006 </td>
        <td>0.020 </td>
        <td>0.010 </td>
        <td>1949.594 </td>
        <td>0.011 </td>
        <td>1769.067  </td>
    </tr>
    <tr>
        <td>UPGAT</td>
        <td>0.037 </td>
        <td>0.108 </td>
        <td>0.226 </td>
        <td>0.098 </td>
        <td>1199.503 </td>
        <td>0.110 </td>
        <td>1060.366  </td>
    </tr>
    <tr>
        <td>Few-shot model</td>
        <td>GMUC</td>
        <td>0.128 </td>
        <td>0.166 </td>
        <td>0.232 </td>
        <td>0.172 </td>
        <td>78.768 </td>
        <td>0.168 </td>
        <td>79.024  </td>
    </tr>
    </tbody>
</table>

##### Filter
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.039 </td>
            <td>0.175 </td>
            <td>0.285 </td>
            <td>0.128 </td>
            <td>1026.465 </td>
            <td>0.138 </td>
            <td>905.267  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.164 </td>
            <td>0.254 </td>
            <td>0.349 </td>
            <td>0.228 </td>
            <td>1740.288 </td>
            <td>0.250 </td>
            <td>1613.610  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.050 </td>
            <td>0.129 </td>
            <td>0.223 </td>
            <td>0.110 </td>
            <td>1101.196 </td>
            <td>0.118 </td>
            <td>968.446  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.086 </td>
            <td>0.223 </td>
            <td>0.340 </td>
            <td>0.175 </td>
            <td>1157.334 </td>
            <td>0.196 </td>
            <td>1084.497  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.078 </td>
            <td>0.178 </td>
            <td>0.296 </td>
            <td>0.151 </td>
            <td>1185.329 </td>
            <td>0.170 </td>
            <td>1077.809  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.037 </td>
            <td>0.167 </td>
            <td>0.288 </td>
            <td>0.125 </td>
            <td>1005.764 </td>
            <td>0.137 </td>
            <td>821.435  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.072 </td>
            <td>0.146 </td>
            <td>0.234 </td>
            <td>0.128 </td>
            <td>1598.388 </td>
            <td>0.118 </td>
            <td>1481.233  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.065 </td>
            <td>0.142 </td>
            <td>0.238 </td>
            <td>0.124 </td>
            <td>1564.019 </td>
            <td>0.138 </td>
            <td>1432.869  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.002 </td>
            <td>0.006 </td>
            <td>0.020 </td>
            <td>0.010 </td>
            <td>1936.439 </td>
            <td>0.012 </td>
            <td>1758.790  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.078 </td>
            <td>0.168 </td>
            <td>0.281 </td>
            <td>0.146 </td>
            <td>1177.974 </td>
            <td>0.165 </td>
            <td>1042.695  </td>
        </tr>
        <tr>
            <td>Few-shot model</td>
            <td>GMUC</td>
            <td>0.128 </td>
            <td>0.167 </td>
            <td>0.232 </td>
            <td>0.173 </td>
            <td>77.695 </td>
            <td>0.171 </td>
            <td>77.926  </td>
        </tr>
    </tbody>
</table>


#### Only high-confidence test data

##### Raw
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.011 </td>
            <td>0.130 </td>
            <td>0.280 </td>
            <td>0.100 </td>
            <td>891.065 </td>
            <td>0.102 </td>
            <td>869.288  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.081 </td>
            <td>0.175 </td>
            <td>0.319 </td>
            <td>0.157 </td>
            <td>1691.038 </td>
            <td>0.160 </td>
            <td>1643.833  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.016 </td>
            <td>0.108 </td>
            <td>0.226 </td>
            <td>0.087 </td>
            <td>957.610 </td>
            <td>0.089 </td>
            <td>931.969  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.058 </td>
            <td>0.146 </td>
            <td>0.303 </td>
            <td>0.134 </td>
            <td>1146.749 </td>
            <td>0.136 </td>
            <td>1110.414  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.057 </td>
            <td>0.141 </td>
            <td>0.281 </td>
            <td>0.128 </td>
            <td>1111.191 </td>
            <td>0.131 </td>
            <td>1085.564  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.010 </td>
            <td>0.131 </td>
            <td>0.289 </td>
            <td>0.101 </td>
            <td>761.389 </td>
            <td>0.104 </td>
            <td>738.024  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.053 </td>
            <td>0.127 </td>
            <td>0.237 </td>
            <td>0.113 </td>
            <td>1515.951 </td>
            <td>0.115 </td>
            <td>1479.857  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.049 </td>
            <td>0.124 </td>
            <td>0.240 </td>
            <td>0.111 </td>
            <td>1483.428 </td>
            <td>0.113 </td>
            <td>1447.763  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.002 </td>
            <td>0.007 </td>
            <td>0.021 </td>
            <td>0.011 </td>
            <td>1719.269 </td>
            <td>0.012 </td>
            <td>1689.562  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.042 </td>
            <td>0.121 </td>
            <td>0.262 </td>
            <td>0.111 </td>
            <td>1082.818 </td>
            <td>0.113 </td>
            <td>1049.431  </td>
        </tr>
        <tr>
            <td>Few-shot model</td>
            <td>GMUC</td>
            <td>0.002 </td>
            <td>0.008 </td>
            <td>0.056 </td>
            <td>0.028 </td>
            <td>96.915 </td>
            <td>0.028 </td>
            <td>96.942  </td>
        </tr>
    </tbody>
</table>

##### Filter
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.017 </td>
            <td>0.188 </td>
            <td>0.317 </td>
            <td>0.125 </td>
            <td>881.970 </td>
            <td>0.128 </td>
            <td>860.370  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.169 </td>
            <td>0.274 </td>
            <td>0.380 </td>
            <td>0.242 </td>
            <td>1678.487 </td>
            <td>0.247 </td>
            <td>1631.426  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.026 </td>
            <td>0.129 </td>
            <td>0.243 </td>
            <td>0.101 </td>
            <td>949.356 </td>
            <td>0.104 </td>
            <td>923.850  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.093 </td>
            <td>0.255 </td>
            <td>0.383 </td>
            <td>0.196 </td>
            <td>1135.032 </td>
            <td>0.200 </td>
            <td>1110.414  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.083 </td>
            <td>0.204 </td>
            <td>0.340 </td>
            <td>0.169 </td>
            <td>1099.705 </td>
            <td>0.172 </td>
            <td>1074.263  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.018 </td>
            <td>0.182 </td>
            <td>0.329 </td>
            <td>0.127 </td>
            <td>752.461 </td>
            <td>0.130 </td>
            <td>729.270  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.074 </td>
            <td>0.164 </td>
            <td>0.266 </td>
            <td>0.139 </td>
            <td>1504.807 </td>
            <td>0.142 </td>
            <td>1468.898  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.064 </td>
            <td>0.160 </td>
            <td>0.271 </td>
            <td>0.135 </td>
            <td>1472.255 </td>
            <td>0.137 </td>
            <td>1436.772  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.002 </td>
            <td>0.007 </td>
            <td>0.022 </td>
            <td>0.011 </td>
            <td>1711.308 </td>
            <td>0.012 </td>
            <td>1681.698  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.094 </td>
            <td>0.196 </td>
            <td>0.323 </td>
            <td>0.170 </td>
            <td>1067.621 </td>
            <td>0.173 </td>
            <td>1034.485  </td>
        </tr>
        <tr>
            <td>Few-shot model</td>
            <td>GMUC</td>
            <td>0.002 </td>
            <td>0.008 </td>
            <td>0.057 </td>
            <td>0.029 </td>
            <td>95.806 </td>
            <td>0.029 </td>
            <td>95.805  </td>
        </tr>
    </tbody>
</table>


## NL27K

### Confidence prediction
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>MSE</th>
            <th>MAE </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="8">Normal model</td>
            <td>BEUrRE</td>
            <td>0.08920 </td>
            <td>0.22194  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.02434 </td>
            <td>0.05176  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.02309 </td>
            <td>0.05107  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.01949 </td>
            <td>0.06253  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.02861 </td>
            <td>0.05967  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.02868 </td>
            <td>0.05966  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.12202 </td>
            <td>0.27065  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.02922 </td>
            <td>0.10107  </td>
        </tr>
        <tr>
            <td rowspan="2">Few-shot model</td>
            <td>GMUC</td>
            <td>0.01300 </td>
            <td>0.08200  </td>
        </tr>
        <tr>
            <td>GMUC+</td>
            <td>0.01300 </td>
            <td>0.08600  </td>
        </tr>
    </tbody>
</table>


### Link prediction

#### All test data

##### Raw
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.086 </td>
            <td>0.237 </td>
            <td>0.358 </td>
            <td>0.186 </td>
            <td>937.320 </td>
            <td>0.212 </td>
            <td>745.028  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.373 </td>
            <td>0.518 </td>
            <td>0.646 </td>
            <td>0.469 </td>
            <td>515.363 </td>
            <td>0.480 </td>
            <td>473.365  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.137 </td>
            <td>0.245 </td>
            <td>0.353 </td>
            <td>0.211 </td>
            <td>2014.542 </td>
            <td>0.232 </td>
            <td>1749.638  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.373 </td>
            <td>0.508 </td>
            <td>0.626 </td>
            <td>0.463 </td>
            <td>222.988 </td>
            <td>0.483 </td>
            <td>204.159  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.369 </td>
            <td>0.502 </td>
            <td>0.622 </td>
            <td>0.458 </td>
            <td>214.093 </td>
            <td>0.478 </td>
            <td>198.758  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.365 </td>
            <td>0.499 </td>
            <td>0.610 </td>
            <td>0.453 </td>
            <td>122.398 </td>
            <td>0.478 </td>
            <td>109.474  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.343 </td>
            <td>0.479 </td>
            <td>0.599 </td>
            <td>0.434 </td>
            <td>253.241 </td>
            <td>0.458 </td>
            <td>226.282  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.347 </td>
            <td>0.477 </td>
            <td>0.599 </td>
            <td>0.435 </td>
            <td>265.915 </td>
            <td>0.459 </td>
            <td>240.235  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.030 </td>
            <td>0.058 </td>
            <td>0.108 </td>
            <td>0.057 </td>
            <td>3022.769 </td>
            <td>0.061 </td>
            <td>2690.496  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.339 </td>
            <td>0.467 </td>
            <td>0.586 </td>
            <td>0.426 </td>
            <td>166.169 </td>
            <td>0.452 </td>
            <td>141.358  </td>
        </tr>
        <tr>
            <td rowspan="2">Few-shot model</td>
            <td>GMUC</td>
            <td>0.287 </td>
            <td>0.409 </td>
            <td>0.536 </td>
            <td>0.375 </td>
            <td>71.484 </td>
            <td>0.375 </td>
            <td>71.447  </td>
        </tr>
        <tr>
            <td>GMUC+</td>
            <td>0.299 </td>
            <td>0.448 </td>
            <td>0.582 </td>
            <td>0.401 </td>
            <td>49.418 </td>
            <td>0.401 </td>
            <td>49.107  </td>
        </tr>
    </tbody>
</table>

##### Filter
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.117 </td>
            <td>0.307 </td>
            <td>0.442 </td>
            <td>0.238 </td>
            <td>874.380 </td>
            <td>0.272 </td>
            <td>683.223  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.761 </td>
            <td>0.884 </td>
            <td>0.946 </td>
            <td>0.829 </td>
            <td>459.719 </td>
            <td>0.849 </td>
            <td>420.153  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.179 </td>
            <td>0.308 </td>
            <td>0.425 </td>
            <td>0.265 </td>
            <td>1957.772 </td>
            <td>0.291 </td>
            <td>1692.880  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.586 </td>
            <td>0.703 </td>
            <td>0.801 </td>
            <td>0.662 </td>
            <td>172.645 </td>
            <td>0.708 </td>
            <td>157.040  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.553 </td>
            <td>0.676 </td>
            <td>0.783 </td>
            <td>0.633 </td>
            <td>163.501 </td>
            <td>0.676 </td>
            <td>151.543  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.580 </td>
            <td>0.706 </td>
            <td>0.782 </td>
            <td>0.656 </td>
            <td>71.203 </td>
            <td>0.715 </td>
            <td>61.337  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.462 </td>
            <td>0.597 </td>
            <td>0.730 </td>
            <td>0.555 </td>
            <td>203.508 </td>
            <td>0.593 </td>
            <td>179.867  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.461 </td>
            <td>0.600 </td>
            <td>0.734 </td>
            <td>0.555 </td>
            <td>216.391 </td>
            <td>0.594 </td>
            <td>194.006  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.031 </td>
            <td>0.062 </td>
            <td>0.113 </td>
            <td>0.060 </td>
            <td>2973.236 </td>
            <td>0.064 </td>
            <td>2641.840  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.530 </td>
            <td>0.654 </td>
            <td>0.765 </td>
            <td>0.611 </td>
            <td>115.004 </td>
            <td>0.658 </td>
            <td>93.692  </td>
        </tr>
        <tr>
            <td rowspan="2">Few-shot model</td>
            <td>GMUC</td>
            <td>0.344 </td>
            <td>0.462 </td>
            <td>0.592 </td>
            <td>0.430 </td>
            <td>67.920 </td>
            <td>0.432 </td>
            <td>67.813  </td>
        </tr>
        <tr>
            <td>GMUC+</td>
            <td>0.371 </td>
            <td>0.505 </td>
            <td>0.638 </td>
            <td>0.463 </td>
            <td>45.874 </td>
            <td>0.465 </td>
            <td>45.495  </td>
        </tr>
    </tbody>
</table>

#### Only high-confidence test data

##### Raw
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.115 </td>
            <td>0.295 </td>
            <td>0.436 </td>
            <td>0.232 </td>
            <td>549.564 </td>
            <td>0.237 </td>
            <td>532.735  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.393 </td>
            <td>0.537 </td>
            <td>0.659 </td>
            <td>0.487 </td>
            <td>436.373 </td>
            <td>0.491 </td>
            <td>430.817  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.168 </td>
            <td>0.287 </td>
            <td>0.407 </td>
            <td>0.250 </td>
            <td>1434.634 </td>
            <td>0.253 </td>
            <td>1435.397  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.405 </td>
            <td>0.544 </td>
            <td>0.662 </td>
            <td>0.496 </td>
            <td>184.430 </td>
            <td>0.502 </td>
            <td>182.857  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.399 </td>
            <td>0.537 </td>
            <td>0.658 </td>
            <td>0.491 </td>
            <td>184.246 </td>
            <td>0.496 </td>
            <td>182.127  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.407 </td>
            <td>0.545 </td>
            <td>0.664 </td>
            <td>0.498 </td>
            <td>97.364 </td>
            <td>0.504 </td>
            <td>95.674  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.382 </td>
            <td>0.522 </td>
            <td>0.645 </td>
            <td>0.475 </td>
            <td>198.500 </td>
            <td>0.481 </td>
            <td>195.908  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.385 </td>
            <td>0.520 </td>
            <td>0.644 </td>
            <td>0.476 </td>
            <td>212.782 </td>
            <td>0.482 </td>
            <td>210.799  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.035 </td>
            <td>0.067 </td>
            <td>0.124 </td>
            <td>0.066 </td>
            <td>2378.456 </td>
            <td>0.066 </td>
            <td>2336.466  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.379 </td>
            <td>0.520 </td>
            <td>0.645 </td>
            <td>0.473 </td>
            <td>114.658 </td>
            <td>0.477 </td>
            <td>113.827  </td>
        </tr>
        <tr>
            <td rowspan="2">Few-shot model</td>
            <td>GMUC</td>
            <td>0.281 </td>
            <td>0.400 </td>
            <td>0.540 </td>
            <td>0.368 </td>
            <td>62.005 </td>
            <td>0.368 </td>
            <td>61.849  </td>
        </tr>
        <tr>
            <td>GMUC+</td>
            <td>0.290 </td>
            <td>0.420 </td>
            <td>0.573 </td>
            <td>0.438 </td>
            <td>45.774 </td>
            <td>0.384 </td>
            <td>49.808  </td>
        </tr>
    </tbody>
</table>

##### Filter
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.156 </td>
            <td>0.385 </td>
            <td>0.543 </td>
            <td>0.299 </td>
            <td>488.051 </td>
            <td>0.306 </td>
            <td>471.784  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.814 </td>
            <td>0.918 </td>
            <td>0.957 </td>
            <td>0.870 </td>
            <td>384.471 </td>
            <td>0.871 </td>
            <td>379.761  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.222 </td>
            <td>0.366 </td>
            <td>0.493 </td>
            <td>0.316 </td>
            <td>1377.564 </td>
            <td>0.319 </td>
            <td>1378.505  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.669 </td>
            <td>0.786 </td>
            <td>0.876 </td>
            <td>0.741 </td>
            <td>138.808 </td>
            <td>0.753 </td>
            <td>138.477  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.627 </td>
            <td>0.754 </td>
            <td>0.856 </td>
            <td>0.707 </td>
            <td>138.781 </td>
            <td>0.717 </td>
            <td>137.864  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.687 </td>
            <td>0.816 </td>
            <td>0.884 </td>
            <td>0.762 </td>
            <td>50.776 </td>
            <td>0.774 </td>
            <td>50.194  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.526 </td>
            <td>0.670 </td>
            <td>0.805 </td>
            <td>0.622 </td>
            <td>153.632 </td>
            <td>0.630 </td>
            <td>152.314  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.525 </td>
            <td>0.673 </td>
            <td>0.812 </td>
            <td>0.623 </td>
            <td>168.029 </td>
            <td>0.632 </td>
            <td>167.344  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.038 </td>
            <td>0.073 </td>
            <td>0.130 </td>
            <td>0.069 </td>
            <td>2329.501 </td>
            <td>0.069 </td>
            <td>2288.222  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.618 </td>
            <td>0.751 </td>
            <td>0.862 </td>
            <td>0.701 </td>
            <td>69.120 </td>
            <td>0.708 </td>
            <td>69.364  </td>
        </tr>
        <tr>
            <td rowspan="2">Few-shot model</td>
            <td>GMUC</td>
            <td>0.335 </td>
            <td>0.465 </td>
            <td>0.592 </td>
            <td>0.425 </td>
            <td>58.312 </td>
            <td>0.426 </td>
            <td>58.097  </td>
        </tr>
        <tr>
            <td>GMUC+</td>
            <td>0.338 </td>
            <td>0.486 </td>
            <td>0.636 </td>
            <td>0.438 </td>
            <td>45.774 </td>
            <td>0.438 </td>
            <td>45.682  </td>
        </tr>
    </tbody>
</table>

## PPI5K

### Confidence prediction
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>MSE</th>
            <th>MAE </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="8">Normal model</td>
            <td>BEUrRE</td>
            <td>0.03662 </td>
            <td>0.15800  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.00346 </td>
            <td>0.02273  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.00336 </td>
            <td>0.02163  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.00350 </td>
            <td>0.02773  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.00357 </td>
            <td>0.02300  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.00356 </td>
            <td>0.02295  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.00769 </td>
            <td>0.05156  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.00319 </td>
            <td>0.02704  </td>
        </tr>
    </tbody>
</table>

### Link prediction

#### All test data

##### Raw
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.000 </td>
            <td>0.025 </td>
            <td>0.137 </td>
            <td>0.053 </td>
            <td>46.498 </td>
            <td>0.059 </td>
            <td>41.783  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.032 </td>
            <td>0.087 </td>
            <td>0.238 </td>
            <td>0.105 </td>
            <td>37.083 </td>
            <td>0.118 </td>
            <td>34.157  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.001 </td>
            <td>0.026 </td>
            <td>0.107 </td>
            <td>0.043 </td>
            <td>197.177 </td>
            <td>0.058 </td>
            <td>141.143  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.031 </td>
            <td>0.081 </td>
            <td>0.228 </td>
            <td>0.102 </td>
            <td>40.756 </td>
            <td>0.127 </td>
            <td>35.730  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.031 </td>
            <td>0.084 </td>
            <td>0.227 </td>
            <td>0.102 </td>
            <td>40.709 </td>
            <td>0.128 </td>
            <td>35.639  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.023 </td>
            <td>0.069 </td>
            <td>0.207 </td>
            <td>0.090 </td>
            <td>42.595 </td>
            <td>0.113 </td>
            <td>35.458  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.029 </td>
            <td>0.082 </td>
            <td>0.234 </td>
            <td>0.101 </td>
            <td>42.224 </td>
            <td>0.125 </td>
            <td>38.202  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.029 </td>
            <td>0.082 </td>
            <td>0.233 </td>
            <td>0.101 </td>
            <td>42.041 </td>
            <td>0.125 </td>
            <td>38.115  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.013 </td>
            <td>0.037 </td>
            <td>0.127 </td>
            <td>0.060 </td>
            <td>85.703 </td>
            <td>0.082 </td>
            <td>63.271  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.034 </td>
            <td>0.088 </td>
            <td>0.236 </td>
            <td>0.106 </td>
            <td>36.820 </td>
            <td>0.132 </td>
            <td>30.896  </td>
        </tr>
    </tbody>
</table>


##### Filter
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.000 </td>
            <td>0.790 </td>
            <td>0.948 </td>
            <td>0.408 </td>
            <td>7.780 </td>
            <td>0.431 </td>
            <td>6.082  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.932 </td>
            <td>0.985 </td>
            <td>0.994 </td>
            <td>0.959 </td>
            <td>4.389 </td>
            <td>0.966 </td>
            <td>4.731  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.013 </td>
            <td>0.150 </td>
            <td>0.313 </td>
            <td>0.121 </td>
            <td>164.695 </td>
            <td>0.179 </td>
            <td>112.057  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.546 </td>
            <td>0.827 </td>
            <td>0.957 </td>
            <td>0.702 </td>
            <td>7.862 </td>
            <td>0.733 </td>
            <td>7.668  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.533 </td>
            <td>0.820 </td>
            <td>0.953 </td>
            <td>0.693 </td>
            <td>7.779 </td>
            <td>0.726 </td>
            <td>7.552  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.268 </td>
            <td>0.597 </td>
            <td>0.821 </td>
            <td>0.470 </td>
            <td>9.671 </td>
            <td>0.521 </td>
            <td>7.334  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.580 </td>
            <td>0.902 </td>
            <td>0.984 </td>
            <td>0.745 </td>
            <td>9.354 </td>
            <td>0.744 </td>
            <td>10.281  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.582 </td>
            <td>0.901 </td>
            <td>0.984 </td>
            <td>0.746 </td>
            <td>9.173 </td>
            <td>0.745 </td>
            <td>10.196  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.232 </td>
            <td>0.398 </td>
            <td>0.624 </td>
            <td>0.359 </td>
            <td>53.152 </td>
            <td>0.418 </td>
            <td>35.539  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.862 </td>
            <td>0.949 </td>
            <td>0.986 </td>
            <td>0.910 </td>
            <td>3.662 </td>
            <td>0.938 </td>
            <td>2.759  </td>
        </tr>
    </tbody>
</table>

#### Only high-confidence test data

##### Raw
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.000 </td>
            <td>0.043 </td>
            <td>0.223 </td>
            <td>0.075 </td>
            <td>27.476 </td>
            <td>0.756 </td>
            <td>27.245  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.060 </td>
            <td>0.155 </td>
            <td>0.400 </td>
            <td>0.166 </td>
            <td>26.964 </td>
            <td>0.166 </td>
            <td>26.863  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.008 </td>
            <td>0.094 </td>
            <td>0.318 </td>
            <td>0.109 </td>
            <td>41.449 </td>
            <td>0.110 </td>
            <td>39.745  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.092 </td>
            <td>0.219 </td>
            <td>0.467 </td>
            <td>0.214 </td>
            <td>23.295 </td>
            <td>0.215 </td>
            <td>22.661  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.091 </td>
            <td>0.217 </td>
            <td>0.460 </td>
            <td>0.211 </td>
            <td>23.743 </td>
            <td>0.213 </td>
            <td>23.105  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.070 </td>
            <td>0.187 </td>
            <td>0.440 </td>
            <td>0.188 </td>
            <td>16.425 </td>
            <td>0.190 </td>
            <td>16.156  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.078 </td>
            <td>0.221 </td>
            <td>0.474 </td>
            <td>0.205 </td>
            <td>29.939 </td>
            <td>0.206 </td>
            <td>29.042  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.080 </td>
            <td>0.219 </td>
            <td>0.473 </td>
            <td>0.206 </td>
            <td>29.963 </td>
            <td>0.208 </td>
            <td>29.070  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.047 </td>
            <td>0.133 </td>
            <td>0.387 </td>
            <td>0.151 </td>
            <td>26.827 </td>
            <td>0.152 </td>
            <td>25.876  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.088 </td>
            <td>0.227 </td>
            <td>0.488 </td>
            <td>0.215 </td>
            <td>15.170 </td>
            <td>0.217 </td>
            <td>14.895  </td>
        </tr>
    </tbody>
</table>

##### Filter
<table>
    <thead>
        <tr>
            <th>Category</th>
            <th>Model</th>
            <th>Hits@1</th>
            <th>Hits@3</th>
            <th>Hits@10</th>
            <th>MRR</th>
            <th>MR</th>
            <th>WMRR</th>
            <th>WMR </th>
        </tr>
    </thead>
    <tbody align="center" valign="center">
        <tr>
            <td rowspan="10">Normal model</td>
            <td>BEUrRE</td>
            <td>0.001 </td>
            <td>0.956 </td>
            <td>0.992 </td>
            <td>0.479 </td>
            <td>2.354 </td>
            <td>0.479 </td>
            <td>2.335  </td>
        </tr>
        <tr>
            <td>FocusE</td>
            <td>0.956 </td>
            <td>0.991 </td>
            <td>0.995 </td>
            <td>0.974 </td>
            <td>7.710 </td>
            <td>0.975 </td>
            <td>7.797  </td>
        </tr>
        <tr>
            <td>GTransE</td>
            <td>0.106 </td>
            <td>0.605 </td>
            <td>0.814 </td>
            <td>0.390 </td>
            <td>24.670 </td>
            <td>0.395 </td>
            <td>23.121  </td>
        </tr>
        <tr>
            <td>PASSLEAF_ComplEx</td>
            <td>0.533 </td>
            <td>0.989 </td>
            <td>0.996 </td>
            <td>0.762 </td>
            <td>8.932 </td>
            <td>0.766 </td>
            <td>8.541  </td>
        </tr>
        <tr>
            <td>PASSLEAF_DistMult</td>
            <td>0.530 </td>
            <td>0.989 </td>
            <td>0.996 </td>
            <td>0.760 </td>
            <td>9.272 </td>
            <td>0.764 </td>
            <td>8.870  </td>
        </tr>
        <tr>
            <td>PASSLEAF_RotatE</td>
            <td>0.318 </td>
            <td>0.940 </td>
            <td>0.989 </td>
            <td>0.638 </td>
            <td>2.034 </td>
            <td>0.643 </td>
            <td>1.994  </td>
        </tr>
        <tr>
            <td>UKGE</td>
            <td>0.402 </td>
            <td>0.992 </td>
            <td>0.996 </td>
            <td>0.696 </td>
            <td>15.925 </td>
            <td>0.698 </td>
            <td>15.276  </td>
        </tr>
        <tr>
            <td>UKGEPSL</td>
            <td>0.404 </td>
            <td>0.992 </td>
            <td>0.996 </td>
            <td>0.698 </td>
            <td>15.961 </td>
            <td>0.700 </td>
            <td>15.314  </td>
        </tr>
        <tr>
            <td>UKGsE</td>
            <td>0.358 </td>
            <td>0.790 </td>
            <td>0.898 </td>
            <td>0.591 </td>
            <td>12.219 </td>
            <td>0.599 </td>
            <td>11.503  </td>
        </tr>
        <tr>
            <td>UPGAT</td>
            <td>0.957 </td>
            <td>0.997 </td>
            <td>0.999 </td>
            <td>0.978 </td>
            <td>1.074 </td>
            <td>0.978 </td>
            <td>1.072  </td>
        </tr>
    </tbody>
</table>