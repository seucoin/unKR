# Results
There are some reproduced model results using unKR



## cn15k

### Raw

|       Model        | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|       BEUrRE       |          yes           | 0.116559997 | 0.282590985 | 0.01100     | 0.13000     | 0.28000     | 0.10000     | 891.06500   | 0.10200     | 869.28800    |
|       BEUrRE       |           no           | 0.116559997 | 0.282590985 | 0.03000     | 0.12700     | 0.24800     | 0.10300     | 1041.15700  | 0.11200     | 916.68900    |
|       FocusE       |          yes           | 83.32640839 | 6.704373837 | 0.08100     | 0.17500     | 0.31900     | 0.15700     | 1691.03800  | 0.16000     | 1643.83300   |
|       FocusE       |           no           | 83.32640839 | 6.704373837 | 0.09000     | 0.16800     | 0.28800     | 0.15400     | 1760.52400  | 0.16800     | 1629.57500   |
|        GMUC        |          yes           | 0.01700     | 0.10900     | 0.00200     | 0.00800     | 0.05600     | 0.02800     | 96.91500    | 0.02800     | 96.94200     |
|        GMUC        |           no           | 0.01700     | 0.10400     | 0.12800     | 0.16600     | 0.23200     | 0.17200     | 78.76800    | 0.16800     | 79.02400     |
|      GTransE       |          yes           | 18.87849    | 4.12786     | 0.015953001 | 0.107601002 | 0.226449996 | 0.086833    | 957.6098633 | 0.089041002 | 931.9691162  |
|      GTransE       |           no           | 18.87849    | 4.12786     | 0.042658001 | 0.112061001 | 0.207380995 | 0.098444998 | 1114.496216 | 0.104622997 | 978.835022   |
| PASSLEAF(ComplEx)  |          yes           | 0.230945006 | 0.399697006 | 0.05800     | 0.14600     | 0.30300     | 0.13400     | 1146.74900  | 0.13600     | 1110.41400   |
| PASSLEAF(ComplEx)  |           no           | 0.230945006 | 0.399697006 | 0.05800     | 0.13100     | 0.26200     | 0.12200     | 1175.95000  | 0.13600     | 1099.12600   |
| PASSLEAF(DistMult) |          yes           | 0.21575     | 0.379233003 | 0.05700     | 0.14100     | 0.28100     | 0.12800     | 1111.19100  | 0.13100     | 1085.56400   |
| PASSLEAF(DistMult) |           no           | 0.21575     | 0.379233003 | 0.05700     | 0.12600     | 0.24400     | 0.11700     | 1203.59800  | 0.13100     | 1092.16200   |
|  PASSLEAF(RotatE)  |          yes           | 0.093751997 | 0.248007998 | 0.01000     | 0.13100     | 0.28900     | 0.10100     | 761.38900   | 0.10400     | 738.02400    |
|  PASSLEAF(RotatE)  |           no           | 0.093751997 | 0.248007998 | 0.02900     | 0.12600     | 0.25200     | 0.10300     | 1020.15400  | 0.11200     | 832.61100    |
|        UKGE        |          yes           | 0.246354997 | 0.408643007 | 0.05300     | 0.12700     | 0.23700     | 0.11300     | 1515.95100  | 0.11500     | 1479.85700   |
|        UKGE        |           no           | 0.246354997 | 0.408643007 | 0.05600     | 0.11600     | 0.20900     | 0.10600     | 1616.04100  | 0.14200     | 1467.33600   |
|     UKGE(PSL)      |          yes           | 0.246304005 | 0.408809006 | 0.04900     | 0.12400     | 0.24000     | 0.11100     | 1483.42800  | 0.11300     | 1447.76300   |
|     UKGE(PSL)      |           no           | 0.246304005 | 0.408809006 | 0.05200     | 0.11400     | 0.21000     | 0.10400     | 1581.75900  | 0.11600     | 1446.81900   |
|       UKGsE        |          yes           | 0.102736004 | 0.255641013 | 0.001595    | 0.00678     | 0.021217    | 0.01128     | 1719.268677 | 0.011713    | 1689.56189   |
|       UKGsE        |           no           | 0.102736004 | 0.255641013 | 0.001503 | 0.00622  | 0.019695999 | 0.01021  | 1949.593506 | 0.011458 | 1769.067261  |
|       UPGAT        |          yes           | 0.14933     | 0.30796     | 0.04200  | 0.12100  | 0.26200     | 0.11100  | 1082.81800  | 0.11300  | 1049.43100   |
|       UPGAT        |           no           | 0.14933     | 0.30796     | 0.03700  | 0.10800  | 0.22600     | 0.09800  | 1199.50300  | 0.11000  | 1060.36600   |


### Filter

|       Model        | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|       BEUrRE       |          yes           | 0.116559997 | 0.282590985 | 0.01700     | 0.18800     | 0.31700     | 0.12500     | 881.97000   | 0.12800     | 860.37000    |
|       BEUrRE       |           no           | 0.116559997 | 0.282590985 | 0.03900     | 0.17500     | 0.28500     | 0.12800     | 1026.46500  | 0.13800     | 905.26700    |
|       FocusE       |          yes           | 83.32640839 | 6.704373837 | 0.16900     | 0.27400     | 0.38000     | 0.24200     | 1678.48700  | 0.24700     | 1631.42600   |
|       FocusE       |           no           | 83.32640839 | 6.704373837 | 0.16400     | 0.25400     | 0.34900     | 0.22800     | 1740.28800  | 0.25000     | 1613.61000   |
|        GMUC        |          yes           | 0.01700     | 0.10900     | 0.00200     | 0.00800     | 0.05700     | 0.02900     | 95.80600    | 0.02900     | 95.80500     |
|        GMUC        |           no           | 0.01700     | 0.10400     | 0.12800     | 0.16700     | 0.23200     | 0.17300     | 77.69500    | 0.17100     | 77.92600     |
|      GTransE       |          yes           | 18.87849    | 4.12786     | 0.025604    | 0.129297003 | 0.243200004 | 0.101469003 | 949.3557739 | 0.104203001 | 923.8496094  |
|      GTransE       |           no           | 18.87849    | 4.12786     | 0.049759001 | 0.129270002 | 0.222671002 | 0.110189997 | 1101.196411 | 0.118147001 | 968.446106   |
| PASSLEAF(ComplEx)  |          yes           | 0.230945006 | 0.399697006 | 0.09300     | 0.25500     | 0.38300     | 0.19600     | 1135.03200  | 0.20000     | 1110.41400   |
| PASSLEAF(ComplEx)  |           no           | 0.230945006 | 0.399697006 | 0.08600     | 0.22300     | 0.34000     | 0.17500     | 1157.33400  | 0.19600     | 1084.49700   |
| PASSLEAF(DistMult) |          yes           | 0.21575     | 0.379233003 | 0.08300     | 0.20400     | 0.34000     | 0.16900     | 1099.70500  | 0.17200     | 1074.26300   |
| PASSLEAF(DistMult) |           no           | 0.21575     | 0.379233003 | 0.07800     | 0.17800     | 0.29600     | 0.15100     | 1185.32900  | 0.17000     | 1077.80900   |
|  PASSLEAF(RotatE)  |          yes           | 0.093751997 | 0.248007998 | 0.01800     | 0.18200     | 0.32900     | 0.12700     | 752.46100   | 0.13000     | 729.27000    |
|  PASSLEAF(RotatE)  |           no           | 0.093751997 | 0.248007998 | 0.03700     | 0.16700     | 0.28800     | 0.12500     | 1005.76400  | 0.13700     | 821.43500    |
|        UKGE        |          yes           | 0.246354997 | 0.408643007 | 0.07400     | 0.16400     | 0.26600     | 0.13900     | 1504.80700  | 0.14200     | 1468.89800   |
|        UKGE        |           no           | 0.246354997 | 0.408643007 | 0.07200     | 0.14600     | 0.23400     | 0.12800     | 1598.38800  | 0.11800     | 1481.23300   |
|     UKGE(PSL)      |          yes           | 0.246304005 | 0.408809006 | 0.06400     | 0.16000     | 0.27100     | 0.13500     | 1472.25500  | 0.13700     | 1436.77200   |
|     UKGE(PSL)      |           no           | 0.246304005 | 0.408809006 | 0.06500     | 0.14200     | 0.23800     | 0.12400     | 1564.01900  | 0.13800     | 1432.86900   |
|       UKGsE        |          yes           | 0.102736004 | 0.255641013 | 0.001595    | 0.007019    | 0.021855    | 0.011495    | 1711.308228 | 0.011936    | 1681.697632  |
|       UKGsE        |           no           | 0.102736004 | 0.255641013 | 0.001503 | 0.006427 | 0.020214999 | 0.0104   | 1936.439209 | 0.011666 | 1758.789795  |
|       UPGAT        |          yes           | 0.14933     | 0.30796     | 0.09400  | 0.19600  | 0.32300     | 0.17000  | 1067.62100  | 0.17300  | 1034.48500   |
|       UPGAT        |           no           | 0.14933     | 0.30796     | 0.07800  | 0.16800  | 0.28100     | 0.14600  | 1177.97400  | 0.16500  | 1042.69500   |


## nl27k

### Raw

|       Model        | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|       BEUrRE       |          yes           | 0.089538999 | 0.222130999 | 0.106187999 | 0.312415004 | 0.454908997 | 0.234743997 | 516.1815186 | 0.239789993 | 500.4590149  |
|       BEUrRE       |           no           | 0.089538999 | 0.222130999 | 0.080660999 | 0.252885997 | 0.377297997 | 0.190451995 | 895.388855  | 0.215119004 | 708.1762085  |
|       FocusE       |          yes           | 290.7572937 | 16.19610023 | 0.387077987 | 0.530840993 | 0.657818019 | 0.482501    | 137.5967255 | 0.486396998 | 136.9906921  |
|       FocusE       |           no           | 290.7572937 | 16.19610023 | 0.368319988 | 0.512754977 | 0.643151999 | 0.464713991 | 176.1968079 | 0.47627601  | 158.1079865  |
|        GMUC        |          yes           | 0.01200     | 0.08200     | 0.28100     | 0.40000     | 0.54000     | 0.36800     | 62.00500    | 0.36800     | 61.84900     |
|        GMUC        |           no           | 0.01300     | 0.08200     | 0.28700     | 0.40900     | 0.53600     | 0.37500     | 71.48400    | 0.37500     | 71.44700     |
|       GMUC+        |          yes           | 0.01500     | 0.10200     | 0.29000     | 0.42000     | 0.57300     | 0.43800     | 45.77400    | 0.38400     | 49.80800     |
|       GMUC+        |           no           | 0.01300     | 0.08600     | 0.29900     | 0.44800     | 0.58200     | 0.40100     | 49.41800    | 0.40100     | 49.10700     |
|      GTransE       |          yes           | 39.83544    | 5.12528     | 0.16800     | 0.28700     | 0.40700     | 0.25000     | 1434.63400  | 0.25300     | 1435.39700   |
|      GTransE       |           no           | 39.83544    | 5.12528     | 0.13674     | 0.24476     | 0.35250     | 0.21145     | 2014.54199  | 0.23173     | 1749.63757   |
| PASSLEAF(ComplEx)  |          yes           | 0.024344999 | 0.051764    | 0.40500     | 0.54400     | 0.66200     | 0.49600     | 184.43000   | 0.50200     | 182.85700    |
| PASSLEAF(ComplEx)  |           no           | 0.024344999 | 0.051764    | 0.37300     | 0.50800     | 0.62600     | 0.46300     | 222.98800   | 0.48300     | 204.15900    |
| PASSLEAF(DistMult) |          yes           | 0.023157001 | 0.051120002 | 0.39900     | 0.53600     | 0.65700     | 0.49000     | 182.90300   | 0.49600     | 180.81300    |
| PASSLEAF(DistMult) |           no           | 0.023157001 | 0.051120002 | 0.36800     | 0.50000     | 0.62100     | 0.45700     | 213.23500   | 0.47700     | 197.71200    |
|  PASSLEAF(RotatE)  |          yes           | 0.015856    | 0.063303001 | 0.39300     | 0.53100     | 0.65000     | 0.48400     | 102.50300   | 0.49100     | 100.45400    |
|  PASSLEAF(RotatE)  |           no           | 0.015856    | 0.063303001 | 0.33400     | 0.46900     | 0.58000     | 0.42300     | 143.79900   | 0.45700     | 122.61800    |
|        UKGE        |          yes           | 0.029072    | 0.059388001 | 0.38900     | 0.52300     | 0.64600     | 0.47900     | 201.85100   | 0.48400     | 199.95100    |
|        UKGE        |           no           | 0.029072    | 0.059388001 | 0.35500     | 0.48600     | 0.60300     | 0.44300     | 251.68200   | 0.46400     | 226.92100    |
|     UKGE(PSL)      |          yes           | 0.028788    | 0.059144001 | 0.38700     | 0.52400     | 0.64200     | 0.47700     | 207.38100   | 0.48300     | 203.61700    |
|     UKGE(PSL)      |           no           | 0.028788    | 0.059144001 | 0.35300     | 0.48500     | 0.60000     | 0.44100     | 252.57700   | 0.46200     | 229.01000    |
|       UKGsE        |          yes           | 0.12202     | 0.27065     | 0.03543     | 0.06695     | 0.12376     | 0.06560     | 2378.45581  | 0.06561     | 2336.46582   |
|       UKGsE        |           no           | 0.12202     | 0.27065     | 0.03000     | 0.05800     | 0.10800     | 0.05700     | 3022.76900  | 0.06100     | 2690.49600   |
|       UPGAT        |          yes           | 0.02922     | 0.10107     | 0.37900     | 0.52000     | 0.64500     | 0.47300     | 114.65800   | 0.47700     | 113.82700    |
|       UPGAT        |           no           | 0.02922     | 0.10107     | 0.33900     | 0.46700     | 0.58600     | 0.42600     | 166.16900   | 0.45200     | 141.35800    |

### Filter

|       Model        | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
|       BEUrRE       |          yes           | 0.089538999 | 0.222130999 | 0.140640005 | 0.407182992 | 0.564805984 | 0.301261991 | 453.1804504 | 0.307830006 | 438.0662231  |
|       BEUrRE       |           no           | 0.089538999 | 0.222130999 | 0.106242001 | 0.325922996 | 0.464942008 | 0.241718993 | 831.166748  | 0.274690986 | 645.0340576  |
|       FocusE       |          yes           | 290.7572937 | 16.19610023 | 0.710326016 | 0.849794984 | 0.930020988 | 0.790171027 | 82.4626236  | 0.793618023 | 82.71473694  |
|       FocusE       |           no           | 290.7572937 | 16.19610023 | 0.662890017 | 0.809248984 | 0.90209502  | 0.748854995 | 117.9376526 | 0.770852983 | 102.0087128  |
|        GMUC        |          yes           | 0.01200     | 0.08200     | 0.33500     | 0.46500     | 0.59200     | 0.42500     | 58.31200    | 0.42600     | 58.09700     |
|        GMUC        |           no           | 0.01300     | 0.08200     | 0.34400     | 0.46200     | 0.59200     | 0.43000     | 67.92000    | 0.43200     | 67.81300     |
|       GMUC+        |          yes           | 0.01500     | 0.10200     | 0.33800     | 0.48600     | 0.63600     | 0.43800     | 45.77400    | 0.43800     | 45.68200     |
|       GMUC+        |           no           | 0.01300     | 0.08600     | 0.37100     | 0.50500     | 0.63800     | 0.46300     | 45.87400    | 0.46500     | 45.49500     |
|      GTransE       |          yes           | 39.83544    | 5.12528     | 0.22200     | 0.36600     | 0.49300     | 0.31600     | 1377.56400  | 0.31900     | 1378.50500   |
|      GTransE       |           no           | 39.83544    | 5.12528     | 0.17914     | 0.30818     | 0.42461     | 0.26475     | 1957.77161  | 0.29136     | 1692.88000   |
| PASSLEAF(ComplEx)  |          yes           | 0.024344999 | 0.051764    | 0.67000     | 0.78600     | 0.87600     | 0.74100     | 138.80800   | 0.75300     | 138.47700    |
| PASSLEAF(ComplEx)  |           no           | 0.024344999 | 0.051764    | 0.58600     | 0.70300     | 0.80100     | 0.66200     | 172.64500   | 0.70800     | 157.04000    |
| PASSLEAF(DistMult) |          yes           | 0.023157001 | 0.051120002 | 0.63000     | 0.75400     | 0.86700     | 0.70900     | 137.31200   | 0.71900     | 136.42900    |
| PASSLEAF(DistMult) |           no           | 0.023157001 | 0.051120002 | 0.55500     | 0.67700     | 0.78400     | 0.63500     | 162.60200   | 0.67800     | 150.42000    |
|  PASSLEAF(RotatE)  |          yes           | 0.015856    | 0.063303001 | 0.66400     | 0.79100     | 0.86800     | 0.74000     | 54.50000    | 0.75400     | 53.55100     |
|  PASSLEAF(RotatE)  |           no           | 0.015856    | 0.063303001 | 0.53700     | 0.66600     | 0.74700     | 0.61600     | 91.49600    | 0.68500     | 73.22600     |
|        UKGE        |          yes           | 0.029072    | 0.059388001 | 0.53200     | 0.68000     | 0.82100     | 0.63000     | 156.86500   | 0.63800     | 156.23600    |
|        UKGE        |           no           | 0.029072    | 0.059388001 | 0.47500     | 0.61000     | 0.74600     | 0.56700     | 201.47700   | 0.60300     | 180.19300    |
|     UKGE(PSL)      |          yes           | 0.028788    | 0.059144001 | 0.53500     | 0.67300     | 0.82100     | 0.62900     | 162.37900   | 0.63700     | 159.88900    |
|     UKGE(PSL)      |           no           | 0.028788    | 0.059144001 | 0.47600     | 0.60400     | 0.74400     | 0.56600     | 202.23200   | 0.60200     | 182.20000    |
|       UKGsE        |          yes           | 0.12202     | 0.27065     | 0.03767     | 0.07310     | 0.13000     | 0.06945     | 2329.50073  | 0.06938     | 2288.22217   |
|       UKGsE        |           no           | 0.12202     | 0.27065     | 0.03100     | 0.06200     | 0.11300     | 0.06000     | 2973.23600  | 0.06400     | 2641.84000   |
|       UPGAT        |          yes           | 0.02922     | 0.10107     | 0.61800     | 0.75100     | 0.86200     | 0.70100     | 69.12000    | 0.70800     | 69.36400     |
|       UPGAT        |           no           | 0.02922     | 0.10107     | 0.53000     | 0.65400     | 0.76500     | 0.61100     | 115.00400   | 0.65800     | 93.69200     |




## ppi5k

### Raw

|       Model        | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| BEUrRE             | yes | 0.036617    | 0.157995    | 0.00000     | 0.04300     | 0.22300     | 0.07500     | 27.47600    | 0.75600     | 27.24500     |
| BEUrRE             | no  | 0.036617    | 0.157995    | 0.00005     | 0.02500     | 0.13700     | 0.05300     | 46.49800    | 0.05900     | 41.78300     |
| FocusE             | yes | 176.9603577 | 13.06495762 | 0.06000     | 0.15500     | 0.40000     | 0.16600     | 26.96400    | 0.16600     | 26.86300     |
| FocusE             | no  | 176.9603577 | 13.06495762 | 0.03200     | 0.08700     | 0.23800     | 0.10500     | 37.08300    | 0.11800     | 34.15700     |
| GTransE            | yes | 4.62883     | 2.03919     | 0.007561    | 0.093762003 | 0.317580014 | 0.109085001 | 41.44915009 | 0.109688997 | 39.74547577  |
| GTransE            | no  | 4.62883     | 2.03919     | 0.000967    | 0.026288999 | 0.106583998 | 0.043067001 | 197.1768036 | 0.058488999 | 141.1428833  |
| PASSLEAF(ComplEx)  | yes | 0.003464    | 0.022728    | 0.09200     | 0.21900     | 0.46700     | 0.21400     | 23.29500    | 0.21500     | 22.66100     |
| PASSLEAF(ComplEx)  | no  | 0.003464    | 0.022728    | 0.03100     | 0.08100     | 0.22800     | 0.10200     | 40.75600    | 0.12700     | 35.73000     |
| PASSLEAF(DistMult) | yes | 0.003358    | 0.021629    | 0.09100     | 0.21700     | 0.46000     | 0.21100     | 23.74300    | 0.21300     | 23.10500     |
| PASSLEAF(DistMult) | no  | 0.003358    | 0.021629    | 0.03100     | 0.08400     | 0.22700     | 0.10200     | 40.70900    | 0.12760     | 35.63900     |
| PASSLEAF(RotatE)   | yes | 0.003499    | 0.027734    | 0.07000     | 0.18700     | 0.44000     | 0.18800     | 16.42500    | 0.19000     | 16.15600     |
| PASSLEAF(RotatE)   | no  | 0.003499    | 0.027734    | 0.02300     | 0.06900     | 0.20700     | 0.09000     | 42.59500    | 0.11300     | 35.45800     |
| UKGE               | yes | 0.003568    | 0.023003001 | 0.07800     | 0.22100     | 0.47400     | 0.20500     | 29.93900    | 0.20600     | 29.04200     |
| UKGE               | no  | 0.003568    | 0.023003001 | 0.02900     | 0.08200     | 0.23400     | 0.10100     | 42.22400    | 0.12500     | 38.20200     |
| UKGE(PSL)          | yes | 0.003561    | 0.022954    | 0.08000     | 0.21900     | 0.47300     | 0.20600     | 29.96300    | 0.20800     | 29.07000     |
| UKGE(PSL)          | no  | 0.003561    | 0.022954    | 0.02900     | 0.08200     | 0.23300     | 0.10100     | 42.04100    | 0.12500     | 38.11500     |
| UKGsE              | yes | 0.00769     | 0.051564999 | 0.046881001 | 0.132703006 | 0.386767    | 0.150641993 | 26.82684326 | 0.152465001 | 25.87584686  |
| UKGsE              | no  | 0.00769     | 0.051564999 | 0.013076    | 0.037477002 | 0.126796007 | 0.060155001 | 85.70326996 | 0.082328998 | 63.27091599  |
| UPGAT              | yes | 0.00319     | 0.02704     | 0.08800     | 0.22700     | 0.48800     | 0.21500     | 15.17000    | 0.21700     | 14.89500     |
| UPGAT              | no  | 0.00319     | 0.02704     | 0.03400     | 0.08800     | 0.23600     | 0.10600     | 36.82000    | 0.13200     | 30.89600     |

### Filter

|       Model        | Confidence Filter(0.7) | MSE         | MAE         | Hits@1      | Hits@3      | Hits@10     | MRR         | MR          | WMRR        | WMR          |
|:------------------:|:----------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:------------:|
| BEUrRE             | yes | 0.036617    | 0.157995    | 0.00100     | 0.95600     | 0.99200     | 0.47900     | 2.35400     | 0.47900     | 2.33500      |
| BEUrRE             | no  | 0.036617    | 0.157995    | 0.00018     | 0.79000     | 0.94800     | 0.40800     | 7.78000     | 0.43100     | 6.08200      |
| FocusE             | yes | 176.9603577 | 13.06495762 | 0.95600     | 0.99100     | 0.99500     | 0.97400     | 7.71000     | 0.97500     | 7.79700      |
| FocusE             | no  | 176.9603577 | 13.06495762 | 0.93200     | 0.98500     | 0.99400     | 0.95900     | 4.38900     | 0.96600     | 4.73100      |
| GTransE            | yes | 4.62883     | 2.03919     | 0.105860002 | 0.605292976 | 0.814366996 | 0.390273988 | 24.6695652  | 0.395209998 | 23.12075806  |
| GTransE            | no  | 4.62883     | 2.03919     | 0.01349     | 0.149678007 | 0.312983006 | 0.121468    | 164.694931  | 0.178894997 | 112.0568085  |
| PASSLEAF(ComplEx)  | yes | 0.003464    | 0.022728    | 0.53300     | 0.98900     | 0.99600     | 0.76200     | 8.93200     | 0.76600     | 8.54100      |
| PASSLEAF(ComplEx)  | no  | 0.003464    | 0.022728    | 0.54600     | 0.82700     | 0.95700     | 0.70200     | 7.86200     | 0.73300     | 7.66800      |
| PASSLEAF(DistMult) | yes | 0.003358    | 0.021629    | 0.53000     | 0.98900     | 0.99600     | 0.76000     | 9.27200     | 0.76400     | 8.87000      |
| PASSLEAF(DistMult) | no  | 0.003358    | 0.021629    | 0.53300     | 0.82000     | 0.95300     | 0.69300     | 7.77900     | 0.72600     | 7.55200      |
| PASSLEAF(RotatE)   | yes | 0.003499    | 0.027734    | 0.31800     | 0.94000     | 0.98900     | 0.63800     | 2.03400     | 0.64300     | 1.99400      |
| PASSLEAF(RotatE)   | no  | 0.003499    | 0.027734    | 0.26800     | 0.59700     | 0.82100     | 0.47000     | 9.67100     | 0.52100     | 7.33400      |
| UKGE               | yes | 0.003568    | 0.023003001 | 0.40200     | 0.99200     | 0.99600     | 0.69600     | 15.92500    | 0.69800     | 15.27600     |
| UKGE               | no  | 0.003568    | 0.023003001 | 0.58000     | 0.90200     | 0.98400     | 0.74500     | 9.35400     | 0.74400     | 10.28100     |
| UKGE(PSL)          | yes | 0.003561    | 0.022954    | 0.40400     | 0.99200     | 0.99600     | 0.69800     | 15.96100    | 0.70000     | 15.31400     |
| UKGE(PSL)          | no  | 0.003561    | 0.022954    | 0.58200     | 0.90100     | 0.98400     | 0.74600     | 9.17300     | 0.74500     | 10.19600     |
| UKGsE              | yes | 0.00769     | 0.051564999 | 0.358411998 | 0.790170014 | 0.898298979 | 0.591023982 | 12.21928215 | 0.598870993 | 11.50269794  |
| UKGsE              | no  | 0.00769     | 0.051564999 | 0.231537998 | 0.398203999 | 0.624309003 | 0.35869801  | 53.1516571  | 0.418422997 | 35.53851318  |
| UPGAT              | yes | 0.00319     | 0.02704     | 0.95700     | 0.99700     | 0.99900     | 0.97800     | 1.07400     | 0.97800     | 1.07200      |
| UPGAT              | no  | 0.00319     | 0.02704     | 0.86200     | 0.94900     | 0.98600     | 0.91000     | 3.66200     | 0.93800     | 2.75900      |