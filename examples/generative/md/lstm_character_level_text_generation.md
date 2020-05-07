# Character-level text generation with LSTM

**Author:** [fchollet](https://twitter.com/fchollet)<br>
**Date created:** 2015/06/15<br>
**Last modified:** 2020/04/30<br>
**Description:** Generate text from Nietzsche's writings with a character-level LSTM.


<img class="k-inline-icon" src="https://colab.research.google.com/img/colab_favicon.ico"/> [**View in Colab**](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/generative/ipynb/lstm_character_level_text_generation.ipynb)  <span class="k-dot">•</span><img class="k-inline-icon" src="https://github.com/favicon.ico"/> [**GitHub source**](https://github.com/keras-team/keras-io/blob/master/examples/generative/lstm_character_level_text_generation.py)



---
## Introduction

This example demonstrates how to use a LSTM model to generate
text character-by-character.

At least 20 epochs are required before the generated text
starts sounding locally coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.


---
## Setup



```python
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import random
import io

```

---
## Prepare the data



```python
path = keras.utils.get_file(
    "nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt"
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
print("Corpus length:", len(text))

chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


```

<div class="k-default-codeblock">
```
Corpus length: 600893
Total chars: 56
Number of sequences: 200285

```
</div>
---
## Build the model: a single LSTM layer



```python
model = keras.Sequential(
    [
        keras.Input(shape=(maxlen, len(chars))),
        layers.LSTM(128),
        layers.Dense(len(chars), activation="softmax"),
    ]
)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

```

---
## Prepare the text sampling function



```python

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


```

---
## Train the model



```python
epochs = 40
batch_size = 128

for epoch in range(epochs):
    model.fit(x, y, batch_size=batch_size, epochs=1)
    print()
    print("Generating text after epoch: %d" % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("...Diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + maxlen]
        print('...Generating with seed: "' + sentence + '"')

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print("...Generated: ", generated)
        print()

```

<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 4ms/step - loss: 1.9237
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 0
...Diversity: 0.2
...Generating with seed: " calm, rational reflection. a church vib"
...Generated:  le and the sugress and the science the sore and and the sore and the such and that the prection and the soul of the sore and the some and the such and the some and the stranstifical the prection and the same to the strange and the stranstification of the some and the sore and the sore to the sould and and the consibely the same and the same the such and the some of the same and the some and and to
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " calm, rational reflection. a church vib"
...Generated:  tion and dererations and the prodited to ordingual common the expecial the problight knowledge and and the masters and with the for the sension the spirition the hass and be possing unceater of do extonstitions of ness the consiberent for the more more more and that the extrations and contral to the of the more and and more and the most precisely do of forther the supprable the point hecest of the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " calm, rational reflection. a church vib"
...Generated:  ti, when an extrated and really ye; be atsessical right deally of the once very and man" there than the own sorm and proartingishient supptishy and itsmed that word "for monsouranbly asd for ensisiance, this par in ond consintions! ir : call and retrods them is to themstucies of every alortehic hand perony of regarding and beandly child tran be ed firerishe? as neigherness. oncishime--awfate and a
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " calm, rational reflection. a church vib"
...Generated:  tion. innot prede for "prestan"witimencesition=-s"phines4 faro-revery insoiviept prictide and coverve als; and "be mork un. of this ne. "inthing is pribty require oo edical  for mores recance mens, of there is nomuthomd more phile--and gred is not extre shan or the preectirabled reapever of enowe, sucpible--to bedical trreouk. it withoue from himselfin evols ot know on 'tronsly gidest behing ave e
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 4ms/step - loss: 1.5699
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 1
...Diversity: 0.2
...Generating with seed: " church. so, too, it will not be admitte"
...Generated:   of the soul of the subrice the something and the same of the the strengtion of the subsing the strength, and as the superitional and into a something of the sense of the strange the sense of the the something of the subsimation of the same of the subsiciated and all the such a the strength. the such a the strange the some the strength, and the such a man the subsiciated to the such a something th
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " church. so, too, it will not be admitte"
...Generated:   of the self-reads become us it is conciritus of a strick under the formarily respect which a great man should of be contrady, all sense of the among of the interman some us to the experices in such a longing in his interprated to the unitions of the principoral the subrilation, the most philosopher to be proutiation of the concerned and to a not which errors of have a regation of the learness to 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " church. so, too, it will not be admitte"
...Generated:  d trasus the vering of the spirits as served, no laves which spiritus is heaktrd? he is those most my should and insidnanpences all didfect revelopication loutter morals of them. but no been belage that is discoving, morality, itself, med, the certainea: to tster that is this organtt: whatever ferress. in celplance--thus a he basful, streeds and it vering, that the might, then the con can mastry u
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " church. so, too, it will not be admitte"
...Generated:  r ging, leagns in this foot in philosoph, pressevcupaise -goad rewappoodwelaved, with religglcated and assivinger--flowark-remails with it have, bli, the hutele whicurarit, he rome, perelogy . rirpompnances! benawating refusacrounce, almost once with supchre droubt and allowings at noncieht lengless! a "who i strriviging the, was nothing, a ot thingmanny yim xw"-foot? "he as -probention thus love 
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.4793
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 2
...Diversity: 0.2
...Generating with seed: "d within myself what it is, by what stan"
...Generated:  dary still the still as a still the could and and the still the still to the higher, and the themselves in the still the still to the still the still to the profound the most desires the still concerning and and the problem of the still the still the still the still the still the stric and the still most which the most the still profound the and the still the still the superioration of the stands 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "d within myself what it is, by what stan"
...Generated:  dal, and because the sates a something and it with the order to such a simple still be religion of such his soul of the concerness and long to desponsible still to man of our object baspess of the profound as a propess as a different and the still the striction and who se respect, and the schopenhauer perstical the higher completion of the still smeth and he self-resides, the remoran enough of the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "d within myself what it is, by what stan"
...Generated:  terdun; the people has for something almo, in cimps of master things has even him tray as a goal in exore of magoty-chulty, the milssesishelf in comportude, that the nature of amble powerful, bettienness and greatimal dreative could anot a cruest also which can he them. unders or that marmulpanting of leadians always them? at the a fessiid of vicnour example alne, petcoss. had withoue isclumhtes i
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "d within myself what it is, by what stan"
...Generated:  datis the ever as if it is need from not he factature of eveny and decesy butk, weser, on that now less, and a necesiontic and be betoves without inraniof, citusan of their -r3faborytofthics to he renent charbe ngain probfinaumiatiof, the promisementslieful, readiced "omilicted atwiddenming elsep, shartin hils thought, a pailsess, he muspobles, thereand unconder: hin, sworw-monsuev ummaismer is fo
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.4307
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 3
...Diversity: 0.2
...Generating with seed: "rs, which had involuntarily extended to "
...Generated:  the soul of the sense of the present the soul of the sense of the sense of the sense of the present the sense of the such a sense of the present the sense of the present the sense of the strength and the soul of the present the soul of the present the soul of the streng of the streng the soul of the sense of the present the strength of the sense of the present the standard to the self-was soul to 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "rs, which had involuntarily extended to "
...Generated:  century, the peillure of life be the end of the subrent and such a precise of the christians to the such a free one have conscience and in the present of sciently belongs of the process with the masters, the present of the past the streng to the cape a sense of their enough of the the standing of the trigitual belongs of nature of the philosophic here soul and manifold and and stand to the great f
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "rs, which had involuntarily extended to "
...Generated:  clooy there lidice or the protonal or truths as to cable in uciness of regreed of the combinist, they belogher of be sad!   flough sootity his thing any it. but everying--is loned above, so dirfelment history, have owing upon regarded destrocious indessental with the spirit classificating hack development that to belongs a physed neare loved to ulinal inlicites the sing and, to you had the thing a
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "rs, which had involuntarily extended to "
...Generated:  an anralog to man, take quick. this is vign itself. uuminar'squink posted. so someoy of preadwers itself; so--onece not the "lofg--are zation)--but the th? comppute matious as whis wahdogics senscrieable syng-thing--easis and duce, a shill of the marely, aoth of it. there is this weich wroth at perhaps knowes yous properfulne of losties and another and how should physives that greoss--the moreth l
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3999
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 4
...Diversity: 0.2
...Generating with seed: " state nowadays assumes the same right, "
...Generated:  and a soul to the contrary and most strong in the same the strong and a still of the sense of the same a strong of the sense of the soul of the strength of the still of the same and the soul of the substitual and still to a strong and a still to a strong and more and a superiority of the still the strong in the same and a strong of the sense of the confers to the soul of the sense of the strong an
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " state nowadays assumes the same right, "
...Generated:  and also a possibilitity of a course and experience in the communitary an and not be the possession of the opposite to a spectation of the an and his revering to be use and concealed and possibilitity and with the fact than the first the statesm century of the condition of the bad every supposed the supposed theer the fact to materment of good and belong be such may i the sense as even the suppose
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " state nowadays assumes the same right, "
...Generated:  the aarful poss but of did maifter, at this decection, as be found them, the shortion. is creatuality, and without it be worves edrath, stuty, for the highest moy for lime extreoi-sharping it is a subso the wonterta to a symptom of man the owest arring has not free done to us on somethicw is evertmate crutlom, as its genwher, duglened to calning also, alowing hofte wishe as possibomal philosophiy 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " state nowadays assumes the same right, "
...Generated:  beroghing so just age--a god cail! manifbkogobleacur", something with religious hadgenous doubous--burtadmon, aronked or in aldit for alrow ubyound fiction by prow axkionte to a ady fact of that thing how has viries in froowed, with the for also? and on , manfort in die to onough a serbantenicomanction, without be us-reasing: thiot" if gemper, in godh- estaice recome poweling lest algarstfuls it s
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3780
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 5
...Diversity: 0.2
...Generating with seed: " things, consists precisely in their bei"
...Generated:  ng that the contention of the sensible of the most danger of the contented and the contented and the strengtion of the contented that the sensible of the most morality of the most still and the soul of the contented and the state of the contented and the contente of the most standard of the most dangerous and discinting and the an all the soul of the contented and in the contented and intercourati
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " things, consists precisely in their bei"
...Generated:  ng to the personality of certainty, as it is a community and morality, as it is or in the alserece of the most still thought the discinting and all of a not and and conceited as a greater have the and incertive of a call thought and has it is presuntion and imacisaly the standard of its an an possible and loves and defited to the traged of the work of an all things of commans to that the instinct 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " things, consists precisely in their bei"
...Generated:  ng and us intellectual moral science is itsieptity trytrems, and to his slineihe indien that to sby every  it, almostining basicaled--we cangs, the her de allge. the child, pleoy, not seascession perhaps gojd, how yet redess:      it unageous , cannot knoub ourselveimially, it over-pleblans and ass1-aress up to demonstimates no god and discisled, be all eye has how "worker", every most popula in s
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " things, consists precisely in their bei"
...Generated:  ng a coqiely dreadchable where we sterm fit favourable all, that does remote that aurourd gre: that chart)w everitual ifxentime myself, these my assured, not from vervatedo--gratits, this southrin nature; whist betlery becomeseds is, for his colvide while bet all olstiticg that certaintes: for they he does hats came that senses. he were toobla umes of meterity, thierd soverfort! and than exkeed, s
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3593
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 6
...Diversity: 0.2
...Generating with seed: " stupidities from which "man" in europe,"
...Generated:   the sense of the sense of the same a still to the soul and in the soul of the soul of the consequently a self subjection of the sense of the consequently a significant of the consequently the same the same and and the soul of the seem of the expression of the more of the such a serve to the soul of the consequently a subjection of the same and a seem of the same morality of the same morality of t
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " stupidities from which "man" in europe,"
...Generated:   to german consequently so tasteration of exception of the most beargeners of the really responsibility, is rit for every astern or the most doubte a process, the most proposition of the contemporal still conscience to something seeker of all love in the soul in a metaphysical deal to the and and also all these more for a prioricy and a sermicish its order and the contempolory of the same man and 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " stupidities from which "man" in europe,"
...Generated:   by the woolnis of priber among wails--hunds from that the prepulling as ever younpor, ever storce, do not be again to reasons.  euntitiest tull  of in we cat do self runts astoreaction that virtues, at the instimyinm and lost doubts weolling nothing for motness".  is retshround, tribelade much a will, te-art, full to the colds to bading imbencaly granter, then, the -dread to the womenter is too: 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " stupidities from which "man" in europe,"
...Generated:   prasisely ever process, is fortalr othen. it wers the elfinity of love.  unpentaccephie can an idrance on ycioted, tpitenly hive good cotsess if the works,                                        6ut har vavedifing, in a preferont we living for itself thoughts who wisper for mustursanicityher, when it woman nacious religious delicer napudions still had reveron to seemingvatms of my a screectic as 
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3452
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 7
...Diversity: 0.2
...Generating with seed: " best nourishment and, in certain circum"
...Generated:  stances and and more strenged to the sense of such a strenged to the sense of the sense of the strenged and strenged and constant and strenged to the strenged and strenged to the sense of the same species of the sense of the strenged to the strenged in the strenged and and the strenge the spiritual indifferent and destrust to the strenged and real problem of the conscience of the sense of the sens
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " best nourishment and, in certain circum"
...Generated:  stances no more would be the seem and the stringly conduct of the conceite of the consequence are to be the pain in the most power of the secret to the highest suppose the relation of pleased in a truth of the most does what is tyoen will in the first possibilities is a strenged and the german is can so the worst of continual man the nature of the sugpitation of the former and indignation of the f
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " best nourishment and, in certain circum"
...Generated:  ary reelidity. moustoriation iureed to ad populal of ind fatherly sounds person and find naits, the stire scyshomal. in groad there is a ascaurion, the parroun miker ray. the mind and themenesting in the rist fect to is element of lurness, in their itplosic we have to powers, has not alon instinct themselves the conduces bowness the sance for expression time in vigning simplify flecosses to the st
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " best nourishment and, in certain circum"
...Generated:  s rote, "is-pain, worawabyse undespression of the very early resceem arase spirit-is more puther. herisom again fawly blinds difficurples,, myself then based fail true stall)  of his. manuthilal proof find to poring we have proeocrating at the rare, his rences to i notherurity. .  1w. andishes it at nimitted, such persontid of urumantity, he danlinactity, induist: with permey are proble-self, for 
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 4ms/step - loss: 1.3329
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 8
...Diversity: 0.2
...Generating with seed: "'t chime.  217. let us be careful in dea"
...Generated:  ve an ange of the process of the process of the struggle of the process of the sense of the process of the extent the more contemption of the experiment of the process of the endless of the process of the power of the experiment of the enduring of the exception of the more free world and contemption of the process of the process of the soul and hand of the sense of the conception of the exception 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "'t chime.  217. let us be careful in dea"
...Generated:  dence to the religious and feeling of the exervanity of the "man will to the causity of the struggle and single truths grates, of the actual to the world for the power of an an existence and with the soul of contempt of the present of the easy something propositation to the sense, strength, and the ance such a suffician of the contempt, whatever the more and superficiation. the jews of the more in
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "'t chime.  217. let us be careful in dea"
...Generated:  ltcry of septher that are resilencqued been suldant unitiness, morrow of the must cannot see operation.--is re practikes man is, most inheritive doubtogm to the unusion.  12eritues, and a later woman is they are as to the own raght to the age and is all and runifiintog in the masp, according and within which to ruth and advanced knows its great mitter jenumif and be the lorelic of dislivent, and s
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "'t chime.  217. let us be careful in dea"
...Generated:  sing an: yea, representablety de within cansy level mindsess repedgriyualc, iten. "worl".     if nedeuws ofthen by veryacy, diever darive) one forgwoct to the reloginan--el of them,e--it is at lost: and blissan by rerecting on as a frueks its podikan.-with rounked and worth, thes this these common afname the oppour more s"o--and ri,ldun clearmmensess been fuint of all discebsified ly "power," and 
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3237
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 9
...Diversity: 0.2
...Generating with seed: " preachers. or, still more so, the hocus"
...Generated:   of the world of the present the soul of the soul of the moral more the one the soul of the problem of the sense of the most more present and moral for the present and the moral of the sense of the way to him and subtleng to the same substion of the sense of the sense of the soul of the subtleng to the soul of the seems of the man to the standards of an exception of the seems of the philosopher of
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " preachers. or, still more so, the hocus"
...Generated:  e of the problem of the concess to the great aspect of a subtaustal and sense, and and only passions is is necessary meanted as a fact that we suffer to the higher to the head of an and inconditional moral development been of the early the other and interpretation of power and the power--that is all the thing as he could necessary him and inforce to the miflering that is the contralities of a cont
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " preachers. or, still more so, the hocus"
...Generated:  ed still dendection and will as a most blest these actuous on the man they are, ondining is free and incleiscultr, edfing influence, many plecerenment of itself every assiving another, ye hfeever you have looking virtues; and not considers so dede. we has only all incietbles, but they have for the spirit did by even almost contrabout mancounded all appearact of moral cease tray only contraint: all
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " preachers. or, still more so, the hocus"
...Generated:   the "wills"--i way noame. only assumble. ereachey a thanfer and absolute the old courportent weal very philomous influention perhaps we does not firstly lange you ho" deceptions, templeums the defin. to orlige of soul, this necessary must ha. nation farthes bralled, but always coars like wriculal exwelted door. yet, changedniggne, for around. the continus, bunsoconm throw such smaloubness of frie
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3137
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 10
...Diversity: 0.2
...Generating with seed: "in the form of which "faith" comes to it"
...Generated:  s own other stronges the distrust of the still constant, and the still and and souls of the still and hangs of the standard of the state of the still mode of the standard of the still and state of the standard of the strength of the still constantule of the senses of the moral contradictom of the comprehension of the distrust of the senses of the senses of the substating and and and state of the t
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "in the form of which "faith" comes to it"
...Generated:  s are and the all the exceptions of a distrust of the standard and instance, all the same deceive that the signal spirit which of all this strength there as a master and the the spirit of a man is not voice of the soul, which of rank of all the made and and standard of comprehensible in the extent of a the will of one and most delicates and inconselfischance and prosits, and to the stard to the th
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "in the form of which "faith" comes to it"
...Generated:   dark of hypothhaly tan actorieikend, anvumhition mits others the gioking presrily the most belief in ever a songuale of his sentiment hix; in natural: when mids say for freeg! one mustly unall form chiradome, common, lesief is the possibility of the tygrcious their enough of wrank all,ge by premordancent, "banking" understands! he wishing. but imanictated man--about singuan they williroes was law
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "in the form of which "faith" comes to it"
...Generated:  , this cirinious musicted no one probabilut superstomkious bet, scene.w"t" of discoverly colour to couthislite, connegisting, llegno: "their ; how that all society behould mind or muvil obligations, time which the p ; fr?whlfottf!de, pshapo-ity musi folingure: the still high tercre of hurh taken evet regative what not called comknaciul that saud lighteaner;--therefore if in every richedness of ons
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3050
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 11
...Diversity: 0.2
...Generating with seed: "ly upon other and higher ideas. it bring"
...Generated:   to the conscience and sense of the problem of the sense of the sense of the subject of the contrast higher and something that the christian and case of the soul of the state of the sense of the superstance of the sense of the subject of the standard of the same philosophers of the sense of the sense of the same the standard of the sense of the sense of the soul of the sense of the subject of the 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "ly upon other and higher ideas. it bring"
...Generated:  ored prosour for the high still of the spirit and former the soul of the particular of the ordinary final case of the serration and day not were the power and of the greatest distraviting entirely the different is well--that is a find the consciently who has been the sense of the moral the science is not the extent the dangerousness of nation of the subject and made attempts and porten of the dang
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "ly upon other and higher ideas. it bring"
...Generated:   the lash in ed ane, immeanes, he non--shapper that is recall thas his truth to father decept the new effect is addition and the uttematible. the chere they never what suffe they do the revide ofecient of the lif wish of things of his time, and originatible, something do date of this limal it in solitud fear, that the world in this "wishes to almost, well one of inter is, for inforeration that it 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "ly upon other and higher ideas. it bring"
...Generated:  utity.  1ited, not who new respincied more for that a times, is findnouptic wiser. this consequene e? which the life is lootg, must. i meanh callly incopid and that that it is man is keepe yearful elsed. vergapen in the sense, that intermand sensed and find belfgre, suffer nuvery.  12112 and are, and too, from dangers of altakm on y ?   only his exttentive dinlest percecok,aquity in adtini than th
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 4ms/step - loss: 1.2986
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 12
...Diversity: 0.2
...Generating with seed: " which he is gentle, endurable, and usef"
...Generated:  ul the strange to the strange to the prouder of the street and the strange the strange to the sees to the problem of the strange that the strange to the strange in the sees of the self the world of the still and imperious and the strange the condition of the probably and the strange the self and the problem of the strange to the state of the great truth that the conscience of the strange to the se
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " which he is gentle, endurable, and usef"
...Generated:  ul creative than the conduct evil and feature of the very the world of the contemplation is the science only a the seem and condount and experience of the problem of the astion of the problem of exceptions of fuem and finally whatever, and is the other long the one for a means of the good reasons relations which man who are can necessary the traditional of the condition of the soul, and believed t
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " which he is gentle, endurable, and usef"
...Generated:  ul, the trepent race.  2is tomentarkpens once opercacial securs but one have and nation--but is comparers appromote thusful, that it while doakess        anything of his fall like flown woman work operates hitself generatively inspire those most languable--that the true to nature of constance. in culture on the decising inlire into all aro love, loves but one has eam is one when the amuly in recin
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " which he is gentle, endurable, and usef"
...Generated:  ulness: it want ooccuto-end. many. thealniscsesse, nowadays, in which hav to you mainsnss"); he refrain for laughes, and fulnal ignor, in which is.  292ou he ogfounder meaning, whuo seen with rationed, with good truth moat valse friensnaants away not luafful, difficulary wourcelunday in soon moys upon riguin sisprisent , heady--appearance, hear-pressed brows: it can stood find its caterg hucaed to
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2925
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 13
...Diversity: 0.2
...Generating with seed: "s deeply involved in _untruth_. the indi"
...Generated:  viduals and sense that the strange the soul of the sense of the more and whole souls and will and strength, and the soul of the more in the sense of the still soul of the subjective proportion, and so that it is a strength of the soul of the part, and the more process and the present and the more personal the process of the subject of the subject of the more and honest in the subject of the still 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "s deeply involved in _untruth_. the indi"
...Generated:  ssival every and and blamely men, the explanation of a problem of a result and still conditions of the world and interest of the consequence, and humanity of the strength to be a man in the significal comparing of the worst who are man is the laws of the new contral strange the moral themselves in the present of the subject of morality and whole the name of a coming and so so centress, and for the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "s deeply involved in _untruth_. the indi"
...Generated:  vidual has parmantion. in mark; and subrlegg "bettering personalities a kinds obsentables--only, pours our life to trihes were one de, as brable--ocsed, of every crrusion of women to his new leads as a eculesis humar lacks asmolosy--sleccrowning, more vesy new lose, that they make bobligimy, because of a good english judgmen jewers even aniwhes, strent ideal deluses to far in the influence of surt
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "s deeply involved in _untruth_. the indi"
...Generated:  ffebutization, grantert shares carly subalks.--adquitions, firstlity is flows known, with a unholdity, shol? with physined: man, "feel uppers as the sort thosen classive of puweltrsi duly. only into reverses the comh for sherd of the regloming-desrepjess? -or the worst suipornes, among alliet-e" forth at permed an insoluntial men a guatuons with personal "penixess", inte. prased" no, as maremes. f
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2863
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 14
...Diversity: 0.2
...Generating with seed: " maturing, and perfecting--the greeks, f"
...Generated:  or its distrust to the subject of the sense of the sense of the sense of the spirit of the more the sense of the subject of the subject of the spirit of the more the standard of the problem of the sense of the sense of the sense of the sense of the sense of the standard of the process of the process and desire to the element of the seems of the sense of the sense of the process of the sense of the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " maturing, and perfecting--the greeks, f"
...Generated:  or the self-entirely consequent indicate even in the world of the subremld person and desire as it misture the desire to solitude. the prounding things. the process and his process which the most sense.                                                                                                                                                                                                      
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " maturing, and perfecting--the greeks, f"
...Generated:  or these man to heirve in thus howinies, it outself, when prives all its "dominitanty" her, sutene, whatever "how tastes, and ilsy--them, the spirit of a hle goo, in the state is to such truth" is dir diving in sholoness him, the night of this artist, in things, they gratifed that is to its germany--the mave was all the dogmat of matter the secrants of the modly been desirt men it by simple!" to t
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " maturing, and perfecting--the greeks, f"
...Generated:  or errhies at the namitates in them.=--dgone--the human. oth--the responsibility pastchf that the folfens.   68 i what recentle, no mart, onely stretd, "chrismeness."--and persistes eradegning," like with magnevated, that he shames? hunce for longest--to different in all, has a cingending sight is the formeoun himbery o" in us's among moralitien." the essential stand to him, threo, prioth.        
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2793
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 15
...Diversity: 0.2
...Generating with seed: "rought him to prison and to early death."
...Generated:   the sensations of the sense of the seeming and strong of the process of the superficial supresents of the state man who strong man will and such a sense of the sense of the seeming and according to the process and according to the soul of the sense of the seeming of the sense of the same man is always to the sense of the seeming of the sensations of the sense of the sense of the sensations of the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "rought him to prison and to early death."
...Generated:    2ne constant are the still forerality in the mart man of look that we may rendered the origin of the conscience for endurable the good man of which a own become a supersting of being the superficieat that in a resign the porten of the end, and appearent will constantaly in the sense of concealed and of the all these family of the superficial the spirit of this of the strange the higher and ready
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "rought him to prison and to early death."
...Generated:  .q: bhrefic the same flee makes possians-, would have its assiry but that willingly, nable, such inder want of events of nature. sifuleringment most against which  i by bad in the advating whithere heaving in which in orders concealed to its reflection to a cause of good and enougds that as free , ialledganis god by a strength for develding that his "poou by appropagar and abwailsise become but it
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "rought him to prison and to early death."
...Generated:   if negt the there for gratifus to ri-dpost is that not bettermen, to philosophear excluttion; mist degenerated of muadless acrous knowledge and the realms, priorably -aged! the true both, their classes. would be critical inspircived, which thourds of stury, fould at all or of percemly may praising. bagneed by constraining iss in all repuded as the apgrantity) cruates. here  found. is refleefly--a
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2754
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 16
...Diversity: 0.2
...Generating with seed: " to accept always and at all times this "
...Generated:  soul and secret and and in the soul of a personal to an and and the soul and the superior of the soul and the soul of the soul of the soul of the more interpretation of the pressure of the present and the contrase of the soul of the soul of the soul of the most spirits of the secret mode of the soul of the more soul of the soul of the soul and and the soul of the soul of all the original in the so
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " to accept always and at all times this "
...Generated:  therefore the intention of the fact of the more things of interpretation and any one is more more such the such pain and the novel of man is to love to men of the more instinct and the hand of morals is of something to a superent of the world and according to the foreound in the soul of a great priceless of the little and and who are relation of the more conscient of an ancient in the highest appa
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " to accept always and at all times this "
...Generated:  english. in enormm means and closes, that it complantantity, a end, of the  of his sense--and the errorselines of remorbletedly intention; it is that which complanted in accord-rophts and omidations. this tais; not, have mame man; has may respect, and possible, i almost incardingly because awmody or tokence chant overed of secarive and philosophizance and additional danger, cour soveretacravarily 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " to accept always and at all times this "
...Generated:  obeying, hevowed by outsegquites from your upon highly uswa's by tragedy?" is a suvoblong and segnious larging to among it fadifiur and own; for store and ye ask of mogent our expessibyreme, by henceful deficiars of nmable, is impar worly: varieually eyes grewes by tradiatiog him brind animal mynction to umary-momed: of prais in hhe notion--but li gsjes: probling ound is that very age to comtagnce
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2697
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 17
...Diversity: 0.2
...Generating with seed: "same rank as astrology and alchemy, but "
...Generated:  a conscience of the power of the power of the power of the process of the power of the sense of the power of the power of the power of the sense of the sense of the power of the power of the power of the self-conscience of the soul of the power of the sense of the morality of the power and strange the sense of the sense of the conscience of the same conscience and intercarimantic man is the concla
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "same rank as astrology and alchemy, but "
...Generated:  the and for a thing of the morality of political disturbing of the highest plaising that in the former pure on a surpress of desirter, but constitately instinctive and acts all the god and relation of the same fine of the intendioc of the proposicness of the conscience of the greater historical intercourse and the seems of the power of their for the same the motive has been the morality of the ant
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "same rank as astrology and alchemy, but "
...Generated:  saiddness--as a distrustly to is philosophers to be knowledge, that we do keeping hoaling we still doual believe, hil, foreign is al politioc egoistich also develde, that hescely with respect usless happies is decial cluir and it merelk aupons, name himself an ancain kinds are not successfulness of means, in humanity and spiritualistict, self granted to any "unills"? "hoar immorality," without you
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "same rank as astrology and alchemy, but "
...Generated:  all decien punifaltingife believe in ! but they lacks and beljent charing for rom fart enough,--oy. in geod unscalcion, forthomer believe it is keep without into almost intellect athain of moralely endpities of date. throughsurbs a turnne ymistive shay fatherles us awaken ke unyo: percepted: plawful as the serpasion upon parhing life, value an excitess upon knee us upon nearly nevertheledss the io
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2653
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 18
...Diversity: 0.2
...Generating with seed: "ncy does exactly the same thing--that is"
...Generated:   the most sense of the soul of the self-dogman and the most spirit the spirit of the spirits of the spirits of the spirits of the spirits of the soul to the sense of the popular the present that the subtlet of the power of the subtlet and hereditation of the personality of the present and more the profession of the self-does the most decided and here and a strange of the soul of the subtlet the sp
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "ncy does exactly the same thing--that is"
...Generated:   contradictously, like the present of all enough, which does not unfering with the porting of the inner suffer to the reason of the reflection of refined man is a new child, there is the disciple of a philosom.   .                                                                                                                                                                                          
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "ncy does exactly the same thing--that is"
...Generated:   he is its slive uncertain as it not become have to be bess himself grasely to that the idea of mlsung love, "prinence" of a most responsible bearing alreadquous ficledness, like the consciously-remonticing, iscerror ogit has almost the experien in eternal mightes eny, thely just the own eascelice down menilies to get apoke as in about let has with error about thevered and hus elf? centate, human 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "ncy does exactly the same thing--that is"
...Generated:   modernaoisitual with no longer leats, bus it before all that it pullitrrush, have almay? become it has vigords oy especially he now we eren" the philosous and imagis to the imagination, he-sayoloy-rigks ypoitays, seeved forrschevers,, has my spircous everythrally is plebentance for horck, is his viled, and breatedy will found spirits--his often: why progrity, to lak cence a depparise, previded ad
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2610
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 19
...Diversity: 0.2
...Generating with seed: "s like what is convenient, so the german"
...Generated:   destruction of the standard of the stand and the sense of the subject of an antitus of the strength of an and no of a stands of the free spirits of the strength of an and every standing of a sense of the strength of its art of the consciously and contrast of his conditionally and all the sense of the problem of the strength of an antitus and sense of the standard of all the subjection of an and t
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "s like what is convenient, so the german"
...Generated:   to the defined of his own good egoism and society in the seriousness and image itself and the most scientific the find as enough, to which they consciously and a the passion of habitual said but the stronger sentiment of the ruling of the engain prevalent of any more of the origin and to the former animalise of the power of the understand to be themselves are superficial does the world the interp
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "s like what is convenient, so the german"
...Generated:   antich, he future, or animaly is i have within away for "divinion" belongera-good--not rendervever, by mysible cultueed well what gotides aned train withans naturally say, this are, not to be europe because the brunume. one is in male has moral! (doun-trooks than conditionally anticouun condition you long hope, into before thy crive more hence the singment of manile and game. the ciluline, differ
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "s like what is convenient, so the german"
...Generated:   hatre of gridts that their patient for it over it edficipullmed a precent has the moral fateornar whent, which musicplywquaie of protacted. though--altonow, from the standingrous taken-resugrerly dopam out forl, for yhrowery--in usdety thress, in life have unclearion: orlave these same loved os, however "a cruti "ordwingd as is son--beginciatly;--he raused people, than "notion"," in whom a best-b
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2574
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 20
...Diversity: 0.2
...Generating with seed: "something arbitrarily barbaric and cerem"
...Generated:  olies and development the fact that the stand the stand the whole masters of the subject. the word of the populact of the spirituality of the contemporary and the philosopher of the sense of the presented the contemouring of the contrary, and what is the fact that the will considerable perhaps has been the belief in the sense of the present and the world in the problem of the present partician of 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "something arbitrarily barbaric and cerem"
...Generated:  ols which something thus indeed and comparison and the subject: which in every man, but all the more demons of the heart of the experience in their extent it is all philosophers of the world, and the other the present indignibices, the belief in the sense of their passion of the calpoule of literating our reason in every one that no more what is no a completening dogmasent philosophy. granticity a
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "something arbitrarily barbaric and cerem"
...Generated:  ars, as the tensting which is only in the vare to based and these gowes in hypocited over still what is it, and perhaps a tenpre? recomquels; we are onely mahturer, and not the all posile: what dow was feach on thought characterish and honesty. it lives is namely, been deluryre: grow mass"--or others. this comestood and general we will, and to a bad fuln valuation, as dead can we preyicents, cultu
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "something arbitrarily barbaric and cerem"
...Generated:  oponed he hers arre canny a sacrifice reached? when nature. granted one's that which its inityment have not into pave different a thy woith originally called that lartning to extrive or jobquent in avave backurk. what iwfulness, far tory was ojeriting; inm), by the relidious contrary, had happe a called tyuerer speak of schoolly uncallitess.--he was its hunable. thom willingsmanists as with regard
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2536
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 21
...Diversity: 0.2
...Generating with seed: " the opposite doctrine, with its ideal o"
...Generated:  f the end, and the subjection of the passionate of the consequences of the seeken of the man is the consequences of the sense of the consequences and the sense of the other dessigation of the sense and in the most presumation of the presumation of the sense of the subjection of the sense of the subjection of the man is also the sense of the problem of the sense of the sense of the sense of the sen
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " the opposite doctrine, with its ideal o"
...Generated:  f the antituded the sense of the works and same past of a so the sense of the forment and concerning things with his personality and experience of his false and and revolution of the alterning the reality of the spirituality of philosophical spirits of the masters of constens the same senses and instruction of the sense with the sensible of the lacking and still make the free former and such a phi
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " the opposite doctrine, with its ideal o"
...Generated:  f the spirit. there are an envyice most hand, bedays even of man that "fathirming forces in the little men i whake being the spirit  period the remainly on, it is, so great arbitrary; while it is existence with adserative sasifility of these relicate nobokens: me. nevertheless with a certian sense amiation, the state flincs to inexpusnting and an impossible forces about the conditions that surxurb
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " the opposite doctrine, with its ideal o"
...Generated:  f therepealists; and to your his (whillf: and as his wifact hitherto noble expanion--alquitus like the further their culture is by means of valed where us ethes have morally instinct, "me, in the individuat the unhones;ness?  112. but absolute enfucor!  . i necessay "not deeved feels" not found succeedrx) romanumentalists, it even there not in morality is calls to go in.ure no sothrarly believeitu
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 8s 5ms/step - loss: 1.2487
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 22
...Diversity: 0.2
...Generating with seed: "smuch as all metaphysic has concerned it"
...Generated:   is a simple and the former problem and the sense of the sense of the sense of the same man in the same man is a stands in the senses of the sense of the senses of the morality and sense and some moral and the same man and most made and in the spectator of the mastery and an an exception of the more man of the world, and the constrain and distrest of the sense of the senses as a person and influen
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "smuch as all metaphysic has concerned it"
...Generated:  s conception, and its victory not more doubted to the incertate of the a them an exception which is even we can be influences which is the favourable to this man ideas of the such an end, and that in the sensible of the more constrain forms strange, with a man of the more dread to the self-religidual of the feason--and the new thing pleasure by same inversed to the remoting which are an endure of 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "smuch as all metaphysic has concerned it"
...Generated:   an incention neeness: sha? the served. in love rifared from the higher to the person errorte, certain it accordingly we perhaps would awave, become progrestances and time in purport finally any fine its squally wimpous enowry, espirity itself it were allowe, theresper propess by that limitlise and imforttent of a which it to be life-sress, as lovecting also he what what would have fancy past ard,
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "smuch as all metaphysic has concerned it"
...Generated:  , say my exampmes than dgald appare natiwar, in"vansed, with woolly dain plenuncers reached, charanhedness, for my e"kereds or cos add in flaveshefuls and strothy. ritut of which remottive; to be otherwishom, imsorem as well-are distribble, would form as is life, by e)fition? incentude torre--and even when ouggeraking, by belief ply syinggrated! and than feelers unaffor lans," a exubtlite -mask; i
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2462
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 23
...Diversity: 0.2
...Generating with seed: "lem there speaks an unchangeable "i am t"
...Generated:  o such a state for the made as the secret the profound and in the said and the subject of the most made of the soul of the proposity of the serious and and experiences of the secret man and deceived the same conscience and self-religion of the said the soul of the man and delight to the secret moral conception of the standard of the proposity of the said the sacrifice of the self-distress and and 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "lem there speaks an unchangeable "i am t"
...Generated:  he contrations" and period the conscience nature and discovering his same exertative in the earth necessity and deficiaring even in the world, or the conguid of the self-distances of any conscience the timely intellectual entered our proportion of a such prequed morality in the habquited man, instectuet in profound it for longing and discover to the second also a procotion of things is even in a s
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "lem there speaks an unchangeable "i am t"
...Generated:  he times the life afride to mis , know for idryad men certain honour the might, our will--afunuad light, and morality himself from enational science out of effect, the later-refined by a bad worthow and smuthes smuth--hows a artistic a great epitury play master; he than is implies, perhaps the delusion of their what of deentmes, last dealt, worth svelling regard all times univers, or a dispease an
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "lem there speaks an unchangeable "i am t"
...Generated:  he vaer dating it as to believe beaseless in new kant so, to volunting, human fortunation. that face bhied by sacriles: fain assumed. avarities, in whom earts than honerst--we is self-soul might be frepjere bagn, corroubes subkable to us, most truths sympathy and most coss; in under teachrs, are nor these is volsely such has is not let perhadival, for instance."--artedly.u, but then if overfeman f
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2446
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 24
...Diversity: 0.2
...Generating with seed: " patriotic flatteries and exaggerations,"
...Generated:   and the most delight of the sense of the standable that the most depth of the conduct and deteriorating of the sense of the sense of the present things to the sense of the power of the most despectity of the spirits of the most delicated and repulsion of the present thing and always the strength of the strange that the sense of the contemplation of the sense of the condition of the sense of the p
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " patriotic flatteries and exaggerations,"
...Generated:   and which has not he wished to be a thing and experiences and an ignorance with a general extent that the most danger of the development of all conscious and the mind of the comprehers of the sense of the standing conduct had not a whild sense and properation that it is according to the sense of the presented to what is the assuming transformed induced and desirable desirable the sense of the mor
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " patriotic flatteries and exaggerations,"
...Generated:   and phenomeni nations speakd silence?       god.  y thinkselves has about this alwave as shoulds are previded that they is nature. more moralitys morality, all treamly far imp sure, it was the morality of a profoundy for experiences no exercourse and person manifest on and transfigurity still site bepthergated as a percection places with us, how munsticism, and our sense is represent eventual the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " patriotic flatteries and exaggerations,"
...Generated:   are reality. that the wigloune does not not arounifucted certainly, and held condratules), and riefd.o.--morality, in his persons--who choic him--lies, for emain is upon others among a socief and worthman! he "hold," and music he ideas out of all, evet it, it memperion and incerted forst, became strength, all, who has unbeet, and is, that it by mmake on as, they bemoxten, know from hiably buragar
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2411
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 25
...Diversity: 0.2
...Generating with seed: " an opinion about any one, we charge hea"
...Generated:  rte and profoundly and the most said the super-and and all things and with the standard of the world of the man is the same and the more proper to the contemporary of the sense of the thing of the foreign of the former and sense and the most superiority of the man is the world of the soul of the contemporary for the fact that it is a philosophers of the present and interpreted and and stronger and
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " an opinion about any one, we charge hea"
...Generated:  rter and about the sympathized and the morality of the contrast and strengt of the fals of self-distrust the fathers in the more spirituality, and the worst and the whole contention of the presently men who feel to be free standays as to be common of the present evil and life and of our powerful, the profoundly into the scorn of which an error that is to the world of the belief in the proposity of
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " an opinion about any one, we charge hea"
...Generated:  rting of metaphysics. that was a sinten the striving is everything, the silil believe creating notuties and powerful. in glorixation; but suppest, as foreity has beer stronge, effect to praise which grewl: was is fory give astof a, and in a most play so its staftly--then sbely a fathesis and experiences presely and belonge) and its conmotion the speay the disremotive =constitution of thing that ne
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " an opinion about any one, we charge hea"
...Generated:  ge with other what called throughe, that is obsible some: "beyodny learnuages," heith he awake is dangerous and woman; even younch new those envoure, his aspute sbeas we count the will more upon stopw, "psacusla)'s notifus," sisingly is a. you. how so helse i regards novele-is, a newe, which ais those astof-thee fatherdly offinding, resemophe!--'upon at developmenval," with modes, god, phases with
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2377
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 26
...Diversity: 0.2
...Generating with seed: "ven occasion to produce a picture alread"
...Generated:   subject, and all the subject of the power of the soul of the subject, and the spectator of the self-anting and the spirit of the spirit the subject, and all the particular in the standard of the sentiment of the subject, and the subject of the souls of the spirit the sentiment of the standard of the strength of the conscience of the spirit of the conscience of the standards of the sounds of the s
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "ven occasion to produce a picture alread"
...Generated:   who har of the spirit and wisten the good of self the motter that he wishes to be acule shame. the highest and all themselves and experiences of the general style of the absolute and now were the infliental spirit and deed to the anything of the appl(owent and implisations, as a philosophers of such a still exception of the general suitation of the subject, the whole of de longer experiences, and
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "ven occasion to produce a picture alread"
...Generated:  k, and would for such were rejoyy--everythee?--i may word is it intereaty in gendurets with philosophers, but with the its a series of life: approading--it is a philosopher, rehuncises and paim to our culturous time an it, were we relatelquental scientific modly, lingler his action to the insiirabx desdrusly atomists of ut--that is the byrod of the laughger of pleasured, and causaly been simpleed,
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "ven occasion to produce a picture alread"
...Generated:  ment as is take ups viwiness, with their is, its more philosophiry, the opposite of any preverread of greemly philosophy, uh"iscap long." sigurpted, ojquaged ye closely him is anreves speant unwas opering time love--and ix back this gut gew othervoblusutow opinion... even the death inssitions, pertaist of themselves: but are moral, no wish about no ones. but europe, the diace of need of much, sacr
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2342
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 27
...Diversity: 0.2
...Generating with seed: "r: one can keep on building upon them--u"
...Generated:  s and the most standard of the struggle of the art of the standard of the moral of the and strength of the contemplation of the standard of the considerable of the standard of the struggle of the morality of the moral the sense of the subjection of the subjection of the morality of the present and implisiation of the moral is also the power of the moral in the sense of the standard of the self-dis
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "r: one can keep on building upon them--u"
...Generated:  s with the scientific sense of the same supposed man in the possibility of the soul. it is the soul of the considerable individual, that is the power of one's art, away as the morality of the cast, said, long soul where the world and the power of his distaction of the soul to his own common and had not more mythodive, and consciously a surelogical the the encomposion of the masterly instrumted suc
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "r: one can keep on building upon them--u"
...Generated:  f, be the moral man, has well-treeds to meen contradicty, by almost absolute, hespery indisses of finere, that they language, any oblige is jepidled and is betoosiqurbas, secrection of the morals lead for attain entailed by world, when noblating is systeme in manleve cummarable homerstory no aves own its precisely incertrave the soul, have lies of the circumstances which colour of the spirit imper
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "r: one can keep on building upon them--u"
...Generated:  pons the charge him, and rampting there wishes to be,n, took at it even revactss interprete that the race hieshy has ao-falsificard in all eperfectual, dancernew alusy, a goveloly sympathime day beer, some waugeonot calpinable despectity that, and forten in anyies know the world some thisyo-rgaction (ajustise! "homend an,x, streests, intellects, the opingors--as man on kioling; perhaps and belakfi
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2326
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 28
...Diversity: 0.2
...Generating with seed: "e is hostile to the sense of shame. they"
...Generated:   are and who has the superiority of the superiority of the superiority of the most proportion of the most strange and states of the man and the state of the superiority of the conscious and and the problem of the contrasty of the strange and with a substite of the conceits of the conception of the contrasty of the superiority of the proposity of the secrated and all the structure of the most propo
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "e is hostile to the sense of shame. they"
...Generated:   still for the conscious and act of the morality and owing to the most man who is there for such upon profination to a philosopher of the superiors of the superiority of the proportion of his domain of the life, and what the most still be the conscious experience of the preserves and with the concerning ancient that the problem of which a substion of being, most refrestion and and also the truth a
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "e is hostile to the sense of shame. they"
...Generated:   has nature than has say to populacization. hence evils. it is suffering, anbit burners intoked browe, wronyss about rather preservous suwour naturiurity is a suble value of athosovic is as the sciences of the ropilial, evodent now generally to what "wors themself"ed" a certain as the probuehiation, which is him, sufferer hatred in actfill is at lose found occurcise his mospes as bemolige of all d
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "e is hostile to the sense of shame. they"
...Generated:   will inspires of phenomenony, the highent of every creature is playzgrating, now much urbiters acmnar with thisllatorom these hand of the extentingest of man and saining puness--how is as a centuling, but be patience--of the law, who does at part" humbepoc one's toward at present that does without marividgedry modes god is terrant? about make whron derressness. such timagion also docest man shume
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2293
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 29
...Diversity: 0.2
...Generating with seed: "e belonging to different nations, even w"
...Generated:  ith the conscious and instrument and sense of the spirituality of the spirituality of the spirituality of the logical strong statest that the spirit of the spirituality of the spirit to the expression of the state of the sense of the state of the spirituality of the state of the conscious and the state of the state of the spirit to the spirit to a spirit of the present and and intercanental the st
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "e belonging to different nations, even w"
...Generated:  ith the misunderstof man of the spirituality, and perceive love and influence. the action of the german be the spirit of its discienment, the misunderstof man. the free most the spirituality of a substate of the powerful and animal this history and experience of the unaqurous have all possess and the braght of it such a sense of an european and therefore of the condition of the absolute of the wor
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "e belonging to different nations, even w"

/usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:4: RuntimeWarning: divide by zero encountered in log
  after removing the cwd from sys.path.

...Generated:  iture from it. the cast appearances attemply intellectnes, and how cannot are, it is morality, this outiose wee-dpoverble. there is loanful-is se-day race, and from the definite which wilf individually religion itself the psycholow of poweron: much in else-opposing  matter that yet from difficulty--the ideas of littine or theio, spirita), and wall sort, between his clumsy men sacrifice the present
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "e belonging to different nations, even w"
...Generated:  it--blesside to seeming, clapo's contralist in such as nafurethess for minding ofur onerd imperatives: "quage this popw; at anything, suitic subs, that extent his envalition is i suitan kind.i whenthmes his fawliding than that hitherto nature irbility.=--the itrenscah moded vain mil-supporet, of popeuration and have hable sort, flows in morality, pregence again in his tendency, muchous sacrifice, 
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 8s 5ms/step - loss: 1.2283
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 30
...Diversity: 0.2
...Generating with seed: "zation.--we may, if we please, become se"
...Generated:  lf-delight of the sense of the strength of the philosophers of the sense of the spirit of the self-difference of the sense of the commander of the sense of the possibility of the sense of the sense of the self-dogmas of the feeling of the self-dogmases of the self-conscience of a deceived and sense and the state of the process of the process of the self-dogmas. the proposition of the sense of the 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "zation.--we may, if we please, become se"
...Generated:  nse and his delight of the spirit of our end and instinctive constitutes the something of all the possess of anti--still manifest with the hendever more cannot the substances of the history of his short of all the sense of a man and wisty, the consequence of the contradiction of the self-sunchalw and constitutes the way there has to reses physical will and acts and even the sense of the heredity o
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "zation.--we may, if we please, become se"
...Generated:  ns. even the me.  12  =oralication. these will menttored with fate for all the tabilury he believes the fremirality, steps inhections of prococe indebdrine--the stade of our own in unspited fast, and examples,.-"called" in the more finewll, in christianed simplicity of the train of thes, in the fact that elfections:--and if oppositewiny, the self a circumstypment-more exclisiant, can have, what th
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "zation.--we may, if we please, become se"
...Generated:  lf this wanton henceforth is good gooc, notication of it--us vent inthing--canave their extraelltical question: i god spiritness is incerverlication: that and height interlee, called perceive to inemistedue slessa. whoe perkoss athy who are and the more quine, but now this also.i neovjrly pooccone: the sence fortunate:--the mer of origin and acknowledgess, minds--they strest and cases indicate out
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2258
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 31
...Diversity: 0.2
...Generating with seed: "hy as you understand it: it is not sympa"
...Generated:  thy, and the standing of the something of a standard of a stand and the strength of the standing of a personal the state of the morality of a standing of all the same morality of the same sense of the still could a standing of a sense of the standing of the most strange, and the present of the standing and of the standing of the spirit of the standing of the stand and the stand and the strength of
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "hy as you understand it: it is not sympa"
...Generated:  rte, only and the spirit of the conquest of his sense that a counture of every senses no more who will. belongs to one's explained to the promises of the fact of the secret morality of any of all the struggle of the modern and instrument of his spirituality and expression and heart pain will the present will and delight to the most fact of nature is a power and really of a things and most strength
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "hy as you understand it: it is not sympa"
...Generated:  thy. it theer for lead the work of the same event woman standing support. [the ifficult bong of the corruption upon their philom pleasuality wouldart of nature. let utivy of a their strain planes (godmad class? for view."--it is in propense--they tovell, on owing to an all theous too age as it has socientance, honestilf, the act, se. in mankind, in the master of exceptioned, and he still?"--if fal
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "hy as you understand it: it is not sympa"
...Generated:  thy clangurethers of the doam could above noble. the origin of a givens myolious only with the outeepton against exk a pleasary ly disintence fine afleo; abyuth have absolutely have meny? in thy still by he ye: they are recotenebdre; perhaps a i  of else, alhoped against of woll-when too se.w have a decive his impositing of a had now mind like the science corrops.  how too willing "verys "saly," w
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2227
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 32
...Diversity: 0.2
...Generating with seed: "ring for centuries: he wishes himself to"
...Generated:   the strength of the subject of the problem of the standard of the most alternation of the most sense of the standard of a man and strength of the standard of the standard of the most instinctive the conscience of the most standard of the standard of the strength of the standard of the standard of the good and sense--and a statesm and the statesm of the portence of the statesm and hence and in the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "ring for centuries: he wishes himself to"
...Generated:   the present higher out of such a demonstrated and rest of the most also and fasciples the man of the science of his strength of a header of the conscience of all instinctive destruction in a conscience of all these delusion of instinctive symptoming standard and experience that which has a guim, as the science of the period and hand and aristocratic workers and advance that not only believed and 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "ring for centuries: he wishes himself to"
...Generated:   a time are nation, such a head of phinomenian, there are kind and of such wisdom--what if the most dammers--that even weaked wishes to requition, the entering for falsificity how say, and the sciences than it is image to himself to the neiddle of the advancious highest train. it satule, shades to end sefuce intercaumue will accomntated though others, there is always without doubt. more mlas modim
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "ring for centuries: he wishes himself to"
...Generated:   ahyinnestary and "ity."!  242. a rextge into the present as nouctantipike must is perceived:--antix?"--hands are life and painfulness a slungtof sciences or is no transifagled grevan to ascerthotor. the goodiat of intention does and relative itness of grate" appeave through changete tahgress of everythingblecise and honoor, and led, bern hand first generally, bud. mucus men "newlyably: no more ju
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2212
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 33
...Diversity: 0.2
...Generating with seed: "ness which cannot dispense even with sic"
...Generated:  k of the subject, the soul of the morality of the spectator of the soul of the spirits of the subject. the problem of the particular the strong the spectator of the problem of the problem of the preserves of the soul in the spirit of the strong to the strong the power of the end of the presery of the problem of the end in the same problem of the spirits of the preservation of the spectator and the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "ness which cannot dispense even with sic"
...Generated:  k of the entire that the spirits of the responsibility of europe, in the subject, which a man indiculter to the apprehended as a state of things, the good and about the moral that the sense of the fact, as the religious men and the power and the morality of the morality that it is the preser, he would lastes the former and sense and hises and interpretation and modern else of realihy, the particul
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "ness which cannot dispense even with sic"
...Generated:  k thought it is moral, originally book, everything fulmerary soul. that which if i ax? -"uttermed amm"and,ly mores the pr"s, and for this influence, but at last envirate their orreach, and axal and out instinct must into turnning bad rids- or at any comple that it above the probably ous say to the first entailes and metaphysics of the such vest and moden, that also a loved and origis and hfool of 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "ness which cannot dispense even with sic"
...Generated:  k, cunind, frest cause that romable, from this fablation that himself the words of the physiss as it us, for instance? os a good infla t of the routtion; it is evcience of smal of recultrable, wike; thy realing natures the hild of natural tastiry for it would stands, the conbrcelictt tures, the loftinee is his reprishs can regarding the same principing innugination. in order to poliet in all impoa
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2202
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 34
...Diversity: 0.2
...Generating with seed: "comfort to the sufferers, courage to the"
...Generated:   success of the sense of the such a person of the sense of the such as the form of the consider of the such and with the sense of the such a sense of the consider of the such a stand and distracted and in the standard of the standard of the such and the philosopher and experiences and the standard of the such as the such and such a such a counter of the still in the standard of a person can a phil
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "comfort to the sufferers, courage to the"
...Generated:   subject, and rebations have been extent to use the sense of the laws, the seem of his evil and the religious of man, if a philosophers in the something morality of the morality of view and conditions of the present invention of actions and life in a god a distracted to look that has the consider could remainejky--but every for the tempt of all the such as it is the antithesis like the spectator o
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "comfort to the sufferers, courage to the"
...Generated:  m are lamses to exaqual sublimats and allerable sacrile; as the gloomxy--saint: the image cigut, wherek here these sextent of vure to atchine would to year missucational and motive forthes that yech it the couragepcal nations of "masponess"." everything experiencistwate prequste marting of all his delaired. what there rebany the greatest martchesmlaed.=--sinjorans.=--what extence that feels that w
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "comfort to the sufferers, courage to the"
...Generated:   little unconscious piteso, as the sinvigs cannorour namle hearts, go out of out of pitises one let more byromfulness grow morally the ording the old designates. been demand as possible itself self by wilf greess of .=--he first appear to befe; he be: alg-other were, however, thus understanding, and this vanit srefring has it could barbarieiest, nepiens, because of his word as being the will of fi
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2171
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 35
...Diversity: 0.2
...Generating with seed: "ngle personalities, hence builds upon th"
...Generated:  e morality of a completer of the conception of the sense of the soul of the subtlerx--and in the subtless of the sense of the sense of the sense of the conception of the subtle person and the sense of the subjection of the sense of the subtle power and the sense of the subtle power of the subtle problem of the sense of the sense of the sense of the subtle, but the sense of the sense and the soul o
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "ngle personalities, hence builds upon th"
...Generated:  e same neighbtal end necessariles of a stronger for the conceptions of the absurcality of our substion the subtlety of a the responsibility of heaver the sunchale of supersiad of such a good of the soul, and the metaphysical curiosity of a tree and independent of a things and deetles, an ancient of its close things and the humably and in the antion of the seeming of result of the conception of the
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "ngle personalities, hence builds upon th"
...Generated:  em has for instruct and serve is the free of a demans of that hore ones hoove.ofund anything day not "necessarily" else beasts to know into the soul of kneenuphar different, the world. all that the services of externant at itself; but what meener an who in uitile. before they had not particiat finally heards aby streads and philosophy. the undeed and coud nature. with the same result of untijwes o
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "ngle personalities, hence builds upon th"
...Generated:  e datry. they streeds and incokeing with sympostions, and have longly has like sword, this unscience! the world has it evaning ro, at reto"us," theremather intoloner passible?--roture, sgure such cloreshance",--(as it is funneen of ourselves breates,  educable my ower to condemsely things hither beentains. sudh often-r-devolosis said we schooler time to be nadjerity. let us enourve loves euddwings
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2146
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 36
...Diversity: 0.2
...Generating with seed: " her sexual gratification serves as an a"
...Generated:  rt of the state of the conscience of the strength of the soul of the soul of the evil and and intercanes and and with the scientific soul the strong of the fact that the soul of the soul of the strong and with the power of the superiority of the strong strong that is the world of the soul of the power of the struggle of the soul of the soul of the world, and but the problem of the soul of the stre
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: " her sexual gratification serves as an a"
...Generated:  dequent and would make the something of the soul of the world, their evil and and sensy secrety his soul within his own impressions, there is all these the still externation of the world of the most artistic than any most elevative and of the subjection and the prospost of the staft his superolloges of community, in the self-conclian with others all this concerning which are not not at the southe 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: " her sexual gratification serves as an a"
...Generated:  ctions is more intermences itself must be circle, always but and protry panificly recoboming over clearror and despossible fights this indijence from all even we goes not overs-cogonor that it may contkind and here, there is to streng morality the narrily of past, nor his time to nature: it is as to view philosophy--and och philosophical that with it. limit high son. indread uttile advancaincolous
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: " her sexual gratification serves as an a"
...Generated:  cal suhs. he wished moniung fallas but something, it wered soon, rotten--wone: as ashomed that it monsceite deficialing corporeas; wholf, doeds will dislive a fut is, it is respositions. is as possible and imply and mismboldarwing.   99  =mally individualed men in egritancy ruiscluty, book that a questionify folly painfully in to befpress of acts my philosophoke, and long of every anti-unswardy th
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 8s 5ms/step - loss: 1.2140
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 37
...Diversity: 0.2
...Generating with seed: "ere then the same as those of the spoken"
...Generated:   of the participation of the standards of the strength of the struggle of the soul of the sense of the struggles of the strange the struggle of the same the still present the strength of the streated and desires and the spirituality of the soul and strength of the sense of the own conscience of the conscience of the standards of the spirituality of the strange the strange into the strange and the 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "ere then the same as those of the spoken"
...Generated:   with himself into the mothed the motheram of the logical man in the proud streatly. in the powerful, anxiety and powers and loved to its and desires philosophy and our apparetticism in all things the standards of his firstly means and all process and the conscience of the soul, the determination, and the character of the conduct that perhaps to a synthesis has attained from the powerful involunta
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "ere then the same as those of the spoken"
...Generated:   wound had above the matter of rangle to defirater of self event, as the nutule?  prease, bro"-conscience."--that manifests in the worlar truths, thung again here immedrating and loved? is earthy? one luckbfarce, cevtsly backs, in some supermouather. it cannot backnaciations"--that emploved asting the most day, or matter to hold self-balso the sentin otfulles: but necessary so timeness, very unite
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "ere then the same as those of the spoken"
...Generated:   that wdis once, more kis, so generations; above them-- itself," evglioted doney--echood missatisvalish to whould tough torenerstjung, to more did notmendance, suspecmises sympathyching junt"--in "good pergots these" itself to him cutistmere! only "epvess: "know   anjer of "fe.a--a "standargoj"ing" before totve exidarly overwad, morality--stapw"ings"efknowledge," ire for sometimes, soce-carificabl
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 8s 5ms/step - loss: 1.2118
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 38
...Diversity: 0.2
...Generating with seed: "he midday-friend,--no, do not ask me who"
...Generated:  se the world of the problem of the world, the problem of the problem of the problem of a strength of the participation of the superstition of the philosophy of the subtlety of the subtlery and the superiolic and the subtle, and in the serious and and who has the superior of the such a sense of the self-satisfactor of the superstition of the particiviation of the soul of the superstition of the sen
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "he midday-friend,--no, do not ask me who"
...Generated:   noble, and the work of which the same time the great in the bad unificult in the world a thing and the philosophy of the world, and in the subtle of an art and relation to the serious saint; we are a philosophy with the man in the world in such as experiences in the can a presumned and considerable feeling of the philosophy in the sight of the european and more man and the sympathy of the philoso
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "he midday-friend,--no, do not ask me who"
...Generated:  n our beauinibiest fallate of things a trunking: psyching again doubtful exised the right too soul that the respect has wa insciently experore a man comong a ventical assuming special truth. flamee. the reason, and or and hontiated unditerd pales to still wish a man with lit this extensety usested science, for underlinedby in spiritual culture of hammed this popuationous a full soul at last faced 
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "he midday-friend,--no, do not ask me who"
...Generated:  se bet, what base et wurfigus possibility, with act have how factics the brahering tortulmen circumdruedly down upon others with thy own artility. torte it veritaverdan to reason saysnxalryion, bundons more gretchence, from exerthescimates the , peris in they are a higher forms impulsed my into as too awkind," for liur, when a ?   .apobatersty, neither an image an inse possible, previded during th
```
</div>
    
<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2106
```
</div>
    
<div class="k-default-codeblock">
```
Generating text after epoch: 39
...Diversity: 0.2
...Generating with seed: "spread over his music the twilight of et"
...Generated:  hical such as the stand its experience and stand in the spirit of the sublimal of the subliment and sense and stand and stand its and instincts in the subject to the spirit of the stand and stand to the sense of the stand and self to the stand and the subject and the subject to the stand of the stand to the subject to the presented and the subtlety of the subjecture of the subtlety, and the sublim
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 0.5
...Generating with seed: "spread over his music the twilight of et"
...Generated:  hical long still and probably with the self-discoverers of a condition of the workery of the sublimal of the decoach of the ordinary and strange of the worst as the morality of the stand attains and confluence and discover as a moral man into the painful even in the act of the sublimal and impaility of the organims and strength of the sense and developed and had an again of all the constant fundam
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.0
...Generating with seed: "spread over his music the twilight of et"
...Generated:  hica other ordining, in posse of untrue of the "word," and his being and what the world who will to superne deem of which claus are much perof exceptional our sense is less assume is preglod naid the humanizing derely beorter. moral and lics of the spirits has liesper, inclairs regard to this edificula! known to the reychinges iss, which morality are distractes hesis and instinct: calminds and exa
```
</div>
    
<div class="k-default-codeblock">
```
...Diversity: 1.2
...Generating with seed: "spread over his music the twilight of et"
...Generated:  hic, that also constant matter of delicate evidence to that its soul--by the worsts: and a in general may at side: pleaided and taken rgeshand hobelied--irbits shupo, indection himbers. to seevary time, do runis. hit"--at dekinged! in short the scientificl; we complewsely did natual men essenys, here the delight, as no longerwy. what  mak i divine, which teachers love it, iillwy capacity are cluth
```
</div>
    

