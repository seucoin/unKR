## generate_psl
A tool for generating softlogic data using PSL rules.

This tool mainly relies on PSL implemented by linqs (https://github.com/linqs/psl). We provide an interface to access PSL to more conveniently generate softlogic data that conforms to the unKR dataset format.

# Guidelines
**Step1** Install pslpython
```
pip install pslpython==2.3.0
```
**Step2** Prepare the dataset

If the dataset you are using is the original dataset (consistent with the format we provided in [nl27k/train.tsv](./data/nl27k/train.tsv)),then you only need to provide the tsv file of samples;
If you are using a dataset that has already been processed into a digital ID format (consistent with the format we provided in [cn15k/train.tsv](./data/cn15k/train.tsv) and [data/ppi5k]((./data/ppi5k/train.tsv))), 
then you also need to provide a relation_id file to map relation names and ids.

The files we provide in the generate_psl/data display the file format more clearly.

**Step3** Set parameters

You can refer to our example to write a yaml file to set parameters.

If the dataset you are using is the original dataset, you can refer to the parameters in [nl27k.yaml](./config/nl27k.yaml). 

If you are using a dataset that has already been processed into a digital ID format, you can refer to [ppi5k.yaml](./config/ppi5k.yaml) and [cn15k.yaml](./config/cn15k.yaml).

You can see the specific explanation of the parameters in [setup_parse](./utils/setup_parse.py).

Specifically, the parameter settings for predict and rule mainly follow the requirements of PSL implemented by LinQS. We will provide an example and a brief explanation here. You can obtain more information from https://psl.linqs.org/.

Size means the number of arguments to the predicate, in the knowledge graph, the size of predicate is usually set to 2. The keyword open indicates that we want to infer some values of this predicate while closed indicates that this predicate is fully observed. You can get more details from https://psl.linqs.org/wiki/Glossary.html.

A predicate example is
```angular2html

  - name: concept_competeswith
  - size: 2
  - closed: False
```

We provide some simple PSL rules examples in [nl27k.yaml](./config/nl27k.yaml), [ppi5k.yaml](./config/ppi5k.yaml) and [cn15k.yaml](./config/cn15k.yaml).
You can refer to more information for rule writing from https://psl.linqs.org/wiki/Rule-Specification.html.

A rule example is
```angular2html
  - binding(P1, P2) & ptmod(P1, P2) -> activation(P1, P2) .
```

**Step4** run infer_psl


Finally, pass your yaml file into the main function of [infer_psl.py](./infer_psl.py) and execute the program to obtain the generated softlogic file. The path to generate the file depends on out_softlogic_filename in yaml.



