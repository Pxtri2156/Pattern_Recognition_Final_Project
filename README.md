# Pattern_Recognition_Final_Project
*This is Pattern Recognition's final project.*  
ID of subject is **CS338**.  
## Topic: **SPEECH EMOTION RECOGNITION**  
## **Definition**:   
We define speech emotion recognition (SER) systems as a collection of methodologies that process and classify speech signals to detect the embedded emotions(refer [here](https://www.sciencedirect.com/science/article/abs/pii/S0167639319302262))
## **Application**
## **Baseline**
## **Datasets**
In this course, we apply three dataset such as MELD, IEMOCAP and URDU for emotion speech recognition project.
### 1.MELD
Resourse: https://github.com/declare-lab/MELD

MELD has more than 1400 dialogues and 13000 utterances from Friends TV series.
Each utterance is labeled by any of seven emotions: Anger, Disgust, Sadness, Joy, Neutral, Surprise and Fear
#### Dataset Distribution
It's author that splits dataset into train, dev, and test.

|          | Train | Dev | Test |
|----------|-------|-----|------|
| Anger    | 1109  | 153 | 345  |
| Disgust  | 271   | 22  | 68   |
| Fear     | 268   | 40  | 50   |
| Joy      | 1743  | 163 | 402  |
| Neutral  | 4710  | 470 | 1256 |
| Sadness  | 683   | 111 | 208  |
| Surprise | 1205  | 150 | 281  |

### 2.IEMOCAP
Resourse: https://sail.usc.edu/iemocap/

There are five sessions in the database (ten actors total). A pair of actor and actress takes responsible for each session.

Each utterance is labeled by any of seven emotions: Angry, Happy, Sad, Neutral, Frustrated, Excited, Fearful, Surprised, Disgusted, Other. Besides, there is label 'xxx' when the number of two lables which anotators label for the utternance is equal

|          | Samples |
|----------|-------|
| Angry    | 1103  | 
| Happy  | 595 |
| Sad    | 1084   | 
| Neutral      | 1708  | 
| Frustrated | 1849 | 
| Excited  |1041  | 
| Fearful| 40  | 
| Surprised| 107  | 
|Disgusted|2  | 
|Other| 3   | 
| xxx| 2507  |

According to table, we just pick labels having the number of samples over 600. So we remove disgusted, fearful, other and xxx because of ratio of 'xxx'label is 50:50
#### Dataset Distribution
We split dataset into train, dev, test with ratio 7:1:2 respectively
|          | Train | Dev | Test |
|----------|-------|-----|------|
| Angry    | 794  | 88 | 221  |
| Excited  | 750   | 83  | 208  |
| Frustrated  | 1331  |148  | 370   |
| Happy      | 428 | 48 | 119 |
| Neutral  |1230  | 137 | 341 |
| Sad  | 780   | 87 | 217 |
### 3.URDU
Resourse: https://github.com/siddiquelatif/URDU-Dataset

URDU dataset contains emotional utterances of Urdu speech gathered from Urdu talk shows. It contains 400 utterances of four basic emotions: Angry, Happy, Neutral, and Emotion. There are 38 speakers (27 male and 11 female).

|          | Samples |
|----------|-------|
| Angery    | 400 | 
| Happy  | 400 |
| Sad    | 400   | 
| Neutral      | 400 |

Because the number of samples for each label is smaller than IEMOCAP and MELD so we decide to split dataset into "tratify" kfold with k is 5. 


## **Features**
## **Methods**
## **Evaluation**
## **Reference**
