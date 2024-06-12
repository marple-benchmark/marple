# Datasheet for dataset "MARPLE"
## Accompanying MARPLE: A Benchmark for Long-Horizon Inference

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

We wanted to evaluate long-horizon inference capabilities using multimodal evidence, such as visual, language, and auditory cues, in simulated household environments. This dataset aims to benchmark AI models' abilities to reconstruct past events and infer agent intents in complex scenarios.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

This dataset was created by researchers from the CogAI and CiCL groups at Stanford University.

### Who funded the creation of the dataset? 

The creation of the dataset was funded by the Stanford Institute for Human-Centered Artificial Intelligence (HAI), National Science Foundation (NSF), and Office of Naval Research Multidisciplinary University Research Initiative (ONR MURI). 

### Any other comments?

## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

The instances represent agent trajectories in simulated household environments where they perform various activities, generating multimodal evidence such as visual, language, and auditory data. The state at each timestep is saved as a .npy and .json file.

### How many instances are there in total (of each type, if appropriate)?

The dataset includes 5000 training instances and 500 testing instances. 

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The instances in the dataset are procedurally generated, and they are chosen to be a representative sample of a larger, infinite set.

### What data does each instance consist of? 

Each instance consists of a pair of agent trajectories. Each agent trajectory is a sequence of observations of the agent's activities in a simulated household environment.

### Is there a label or target associated with each instance?

Yes, each instance includes a label.

### Is any information missing from individual instances?

No.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

Yes, through the directory filestructure (agent trajectories in the same room, agent trajectories for the same inference scenario, etc.)

### Are there recommended data splits (e.g., training, development/validation, testing)?

Yes, the dataset includes training and testing splits to evaluate model performance.

### Are there any errors, sources of noise, or redundancies in the dataset?

The authors are not aware of errors or redundancies.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

The dataset is self-contained.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

No.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No.

### Does the dataset relate to people? 

No.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

N/A

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

N/A

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

N/A

### Any other comments?

## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

The data was generated using a simulated GridWorld environment, with procedural generation of diverse agent behaviors and multimodal evidence. Please see the supplementary detail of our paper for more details.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

The data was generated using software, namely our codebase and scripts.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

The dataset was procedurally generated with random sampling.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

The paper authors collected the dataset.

### Over what timeframe was the data collected?

The dataset was generated over five days during February 2024.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

N/A

### Does the dataset relate to people?

N/A

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

N/A

### Were the individuals in question notified about the data collection?

N/A

### Did the individuals in question consent to the collection and use of their data?

N/A

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

N/A

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

N/A

### Any other comments?

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

Yes, we removed generated instances that were incomplete.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

No, as they were incomplete instances and could not be used for evaluating inference abilities.

### Is the software used to preprocess/clean/label the instances available?

Yes, we provide our data generation scripts in our codebase.

### Any other comments?

## Uses

_These questions are intended to encourage dataset creators to reflect on the tasks
for which the dataset should and should not be used. By explicitly highlighting these tasks,
dataset creators can help dataset consumers to make informed decisions, thereby avoiding
potential risks or harms._

### Has the dataset been used for any tasks already?

Yes, it has been used to evaluate long-horizon inference capabilities of AI models in a proposed inference task.

### Is there a repository that links to any or all papers or systems that use the dataset?

Our paper is under review, but the data repository will provide links to papers that use the dataset in the future.

### What (other) tasks could the dataset be used for?

The dataset can be used for a variety of tasks related to multimodal inference, event reconstruction, and high-level reasoning in AI models. It can also be used for cognitive science research to investigate human-like reasoning processes and understand how humans infer actions and intentions from multimodal stimuli.

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

The dataset is composed of procedurally generated GridWorld environments, which may limit its application to real-world scenarios that require physical understanding and low-level reasoning.


### Are there tasks for which the dataset should not be used?

The dataset should not be used for low-level physical reasoning or tasks requiring realistic physics simulation. Instead, it should be used for high-level, semantic reasoning.


### Any other comments?

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

The dataset will be freely available for public download.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

Our dataset is hosted on Github.

### When will the dataset be distributed?
N/A

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The dataset will be distributed under the CC-BY-4.0 license.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

### Any other comments?

## Maintenance

_These questions are intended to encourage dataset creators to plan for dataset maintenance
and communicate this plan with dataset consumers._

### Who is supporting/hosting/maintaining the dataset?

N/A

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

The authors can be contacted at emilyjin@staford.edu.

### Is there an erratum?

N/A

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

Yes, in the event that errors are found, the dataset will be uploaded as a new version at the same location.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

N/A

### Will older versions of the dataset continue to be supported/hosted/maintained?

Yes, they will be hosted as previous versions on the Data Repository.

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

At the moment, there is no mechanism to build on to the dataset. However, in the future we may release new expanded versions of the dataset with opportunities for contribution.

### Any other comments?
