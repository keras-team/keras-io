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
import sys
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
print("corpus length:", len(text))

chars = sorted(list(set(text)))
print("total chars:", len(chars))
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
print("nb sequences:", len(sentences))

print("Vectorization...")
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


```

<div class="k-default-codeblock">
```
corpus length: 600893
total chars: 57
nb sequences: 200285
Vectorization...

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

    print("Generating text after epoch: %d" % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("----- diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

```

<div class="k-default-codeblock">
```
1565/1565 [==============================] - 7s 5ms/step - loss: 1.9779
Generating text after epoch: 0
----- diversity: 0.2
----- Generating with seed: " been taken seriously from the first--he"
 been taken seriously from the first--here and every the present in the sense of the some the sense the sense of the sentic and more of the most to the some the subtertion the sense of the soul of the senses of the sense and consciented to man of the sense of the sense and instinct of the most preger and spection of the sense of the present to the sense of the some the sense of the sense of the sense of the senses of the sense the some 
----- diversity: 0.5
----- Generating with seed: " been taken seriously from the first--he"
 been taken seriously from the first--here they with the sents the lofes and the sentity of the sole nothing would and the breating the morally the sensess by not in the spresed and into the senting the most counted indiduations by the some the but the sense and pase of the some the sociented to for the prepersed by the priget by the plain and every of the most with the proble of the sente and manbing that which the understing is belong
----- diversity: 1.0
----- Generating with seed: " been taken seriously from the first--he"
 been taken seriously from the first--he would, and
intortolf; and will tenders, the har its good
rane of retoloun as hitsergasionity: which lone--rase is most is all mavect.--which religionsd is marcy. 
for the kne-nex peressejong evanile froon
xerlactical sensited to langerhaval---repases to the old de(ifle! man, in genera)ityed." we earess powears, we awnorgever that
the conkence
to
an hathernt" his xigming plasely peris abone every 
----- diversity: 1.2
----- Generating with seed: " been taken seriously from the first--he"
 been taken seriously from the first--here22); the
srinct theme
standigt lems whith
wrother hinsoxireved, regardinate atolked, nolers, when ssefutnived the blitt, and feeter wall cowly bosk
sociectic dingucess popan mylilar! paves slens, michthal
infllastius and revoling,
rewlouts.--theysy--in which, is [brines.hed was xersg powerans that revell kempts
bwost has notomi gpeal origit, and
e yourd thpe
do modect, and narvacion, to might
no
1565/1565 [==============================] - 7s 5ms/step - loss: 1.6221
Generating text after epoch: 1
----- diversity: 0.2
----- Generating with seed: " like) can be brought
to bear upon him. "
 like) can be brought
to bear upon him. the something it is a press of the sense of the something that it is a such and a did the sense of the press of the conseranity and it is the self-down and a subsinders and a things of the something of the presertant the self-does and it is a subterion and a the something and a such and a such and a sense, and it is a such a such and a such and a speak and it is a such and and a succession of the 
----- diversity: 0.5
----- Generating with seed: " like) can be brought
to bear upon him. "
 like) can be brought
to bear upon him. is the higher at the mankind of the refutional conceral pressuch as the expersental and a presses and it is something a not and love and any all that is the pressional in itself and the such a man the nature and it could is courable of the something of the subtle the tranifical science and a suppre sense and the become the first it is the effect, and he seem the conscious and a something and somet
----- diversity: 1.0
----- Generating with seed: " like) can be brought
to bear upon him. "
 like) can be brought
to bear upon him. has being not as heiskly to praie who sispay them, no because a nimated se sension of leseness--that be "many. scendure and is creas at a to be that it pasten its oal sens, his thought it who an abtry now spete of this an all akeful must many
as they distrangest
good it be not
goe ngwands its out ate the dered thinkind of "semmoment.
```
</div>
    
<div class="k-default-codeblock">
```
1cmsal it. thick in e?e is need the
revalle and the more im is 
----- diversity: 1.2
----- Generating with seed: " like) can be brought
to bear upon him. "
 like) can be brought
to bear upon him. a bass even bast to mividual
manifineing.
```
</div>
    
<div class="k-default-codeblock">
```
lvevar still bigde humanners of
stepts have rin of that yeer oftreats xhan
stience another his .u
yt
an aptand?,
in the
ply who anctors of with it is necessive dure
detem. -it ear somethionhes of the lamid:--he wan be curable, but absonism to the poss; is
question (of his owt our other abweecial go theme8?
-gree
and anttrant-will prace of socuation; that 
1565/1565 [==============================] - 7s 5ms/step - loss: 1.5337
Generating text after epoch: 2
----- diversity: 0.2
----- Generating with seed: "we may, if we please, become sensible, e"
we may, if we please, become sensible, ever the present the standard of the sense and and and an and the morality of the persinted to the same the procosity, the said that is the partiction of the man is all the power and all the sainty, the proportic and self-conceptions and and in the power of the sacrifice and the self-distrust of the here the power and the sense, and the profound and the serroced to the consideration of the power th
----- diversity: 0.5
----- Generating with seed: "we may, if we please, become sensible, e"
we may, if we please, become sensible, every the person--and german in considerantly. an an ever the portular to him, when the self-desire one is reading from and have not a periliss impartious to the future there are the religious of the right that the had ever the procosition in the conception, as the sublime and are the religious life the a certain feelings and serror and simple, which was not as in a profound the same, which is a me
----- diversity: 1.0
----- Generating with seed: "we may, if we please, become sensible, e"
we may, if we please, become sensible, emagenly moralsy, and comparation. what unitior--he inflace afrapectine-supal liffer hardure
ethic, a ducanded tthe casion. on this lest this is
find of culve, sace? fut us him. let once! morly and reinsice of "from the jewarche they were be perhaps not germany cans-ners refiine upon the
proposing, such, he sancule
indeling doubs the will loses
us; the tmuch prope--; aly dine-reass: while place hen
----- diversity: 1.2
----- Generating with seed: "we may, if we please, become sensible, e"
we may, if we please, become sensible, every conditurallable enolmely one all a indo: would no thoreed thae isual
langishen, that tybe everfounds irde---are
pack indeesse equisuble
oud and seest, martistar even
strengen nature.--a goven,
le, even hellidgelf-indiny in upitsot wanted, -itstrayard discristic
speces and beliesscly, foroure and docme,. it isbodk, not do alunts in lackiaring to
be only
dor that prystingruple ;over overnacorsh
1565/1565 [==============================] - 7s 5ms/step - loss: 1.4857
Generating text after epoch: 3
----- diversity: 0.2
----- Generating with seed: "riness in which they are
steeped by thei"
riness in which they are
steeped by their problem of the conditional state of the belief in the problem of the belief in the end in the herd to the problem of the soul to the problem of the problem of the belief in the herd the problem of the problem of the problem and the state of the problem of the same moral terrs of the morality and the problem of the strange the moral endurities of the problem of the conscience of the end in the so
----- diversity: 0.5
----- Generating with seed: "riness in which they are
steeped by thei"
riness in which they are
steeped by their has experience a stain conscience mankind have action, the lacking of the partion of him are is a character of the problem of the believe and action of was in the same sciente, has now the cases this out of trans the morality of the process of the casterial sympathy is the believed itself in which it is the profound herd the help to the probably in are metaphysic to stance, the cases the convict
----- diversity: 1.0
----- Generating with seed: "riness in which they are
steeped by thei"
riness in which they are
steeped by their properds
of the him the
me..--who had
not which is we sould, pnesutips of god in our learnian from
happered out--it is libet to continuthssionally virtue worth if a times of-puipost car world is lives that the southing atternity chase soughty in of,"" to somenction at scapt to
dograge of the deapsed strong wobl have are noble
deepated nimalitating schet for alneement instinctionsly hatine mare-s
----- diversity: 1.2
----- Generating with seed: "riness in which they are
steeped by thei"
riness in which they are
steeped by their,, bidl--the wond proima.y new perceptions the trannlicing san to religion, from
their lifeity with--he isme fol )(  for cassible" and calls of womlan," i rajoosing luksuctmenly
such
dombirol
certainty, mother (and disposeration there astious moraits, which have not the soneble
hun ade every haurd asterd kindly; necessary
moral stranderness with freets, phaligis of accomplumfe. oursly oneit regar
1565/1565 [==============================] - 7s 5ms/step - loss: 1.4554
Generating text after epoch: 4
----- diversity: 0.2
----- Generating with seed: "ho would have time to wait for such
serv"
ho would have time to wait for such
servantion of the most superiors of the preservations of the feelings of the more of the most such and the strenge of the sense of the sense of the commanded in the sense of the more of the sense of the most sense, and the most sense and a such and and a result of the art of the more the present the more conscience and and the soul of all the more of the sensthe same soul o
----- diversity: 0.5
----- Generating with seed: "ntment have been made against pictures o"
ntment have been made against pictures of any strong the strong possible of the man, as the love of missumacians in world of a truth" who what it is there are there is a territetion of the soul of the or a degrees of the thought in a contemptible, and the reverence of by the despised and preciselfully in the interpreted and developbles of his
specially imperiority is the ancient reason of morals which will more imperfog that there are r
----- diversity: 1.0
----- Generating with seed: "ntment have been made against pictures o"
ntment have been made against pictures of aitre (when there is forgoured himself finerequing thereby of his la culse for every worde-for riliage or
withd--chostralom, diversting, what these deceutor is no
least growthed, sayin remost 
comprehfineres in the
powerfles to deast of judgment, and beoutelus, "smovef; what myre spicitated upon of their middain that there it is amoners of the free beings against ordinared to exubrace an and exi
----- diversity: 1.2
----- Generating with seed: "ntment have been made against pictures o"
ntment have been made against pictures often seems to thou, wiendly
only their syinen immenses and presubranmedical
suiming , moders) for ourselve heours itself, but which even nature, a defendity. the european, persontol, be! any reudincent" froutury, said, that more? in 
suman:--whse?t of it, or rank, one, to nature
stainism,
evol.wdocranings--for
the crided of slifoles, depect destritates for also progriast
woses are a certainly that
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3648
Generating text after epoch: 11
----- diversity: 0.2
----- Generating with seed: "
world merely to put an end to the numbe"
```
</div>
    
<div class="k-default-codeblock">
```
world merely to put an end to the number to the sense of the soul, and a consciousness, and the same and the same fine the sublime and indeed to the stake and the same and the same and interest and developed and indeed to himself and in the soul and an and and and in the soul and consciousness, and the consciousness and consequences and individual and individuals and individual and independent of the same time and and in the soul in th
----- diversity: 0.5
----- Generating with seed: "
world merely to put an end to the numbe"
```
</div>
    
<div class="k-default-codeblock">
```
world merely to put an end to the number to the destructive and more and independent even in the highest in the stare an individual hating to all the partical call science is as a possible and definitions and delight and .
experional motives is not even to the an ancient and in the process, which appropriate and when i meante of the in itself and master is the else death and process
of the basis in short, and the assight for the sight 
----- diversity: 1.0
----- Generating with seed: "
world merely to put an end to the numbe"
```
</div>
    
<div class="k-default-codeblock">
```
world merely to put an end to the numbers! hand!
    live ebacly
that it has the best gain of a go out of its sympathy; i peises and
impoint of
part
higher swept.
a
pensition the point
we forinantly, mavy, the discovery is, as now reverences, inceroned, symocran and suspicion and also-proposition at the dreamful, now metrey and herely.
```
</div>
    
<div class="k-default-codeblock">
```
2t. hese and direces venit if recognct, as the constinite
how should be by
the
ability; on it find i
----- diversity: 1.2
----- Generating with seed: "
world merely to put an end to the numbe"
```
</div>
    
<div class="k-default-codeblock">
```
world merely to put an end to the number.
```
</div>
    
<div class="k-default-codeblock">
```
esa namely, out of frank as is gars
beg cholds, and
peind onty it as at all ovidian p to xunk, a self-jucting to alast--incalled sinism setracus began its chara his any image is learn whighes in to wheme, (that in
the
highest
of the rys. but
it is only! uppere regarming inscinuuses, and the badd dissib?quist habse--he
know patter and
exinceded and here
that on this. not their
tabities himselit
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3584
Generating text after epoch: 12
----- diversity: 0.2
----- Generating with seed: " religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however gra"
 religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however grantes of the same standard of the same time with which he will be a still and such a spiritual and a surmition of the case of the standing of the same time of the same time of the secretable to the sacrifice and all the sacrifice to a sided and deceived to the soul of the sacrifice and desire and self-conscience the self-subtle and in the same fact that is the same signs of the same to the standing
----- diversity: 0.5
----- Generating with seed: " religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however gra"
 religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however grantes
would alwand and desire of success of the conditional form of glorification aften will more who say to the thing as the intellectual her it is so who could be the long to be standing a moral and existence, and and his such suffering, in the secret as the will, and the thing of this being of alled to the existence to the world is to problem of god which we discoming, and is at its action and i
----- diversity: 1.0
----- Generating with seed: " religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however gra"
 religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however graying his own gloved is the soun of every teurs to posteriously, eithers who saysually torted in, should for itself", something
of the gemanr! but not as they have a phact, such as now sumily and
most tamo, self-dissid among honefille, has manla! are rewagenessess, in that a ideally. supposing in
the soul detained his indimiblene, and , first love enough,
and folly (and feeling in
society, of look-
----- diversity: 1.2
----- Generating with seed: " religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however gra"
 religion of sympathy.
```
</div>
    
<div class="k-default-codeblock">
```
207. however grains, but name the
anwarigatiut.=--how "been vained church metaphysicism and confegsing, ascetic
```
</div>
    
<div class="k-default-codeblock">
```
          dismentation that events human" ofly inrin livelfuracniniss amongly
that a pysyitier oneself destedial, upon
the pious leard, the sile ebve-distrustorously
good self, from this nature--indocabless to also, as almost to soveces;--seemon that is
ityl-werpet naw powerful epissly
do
not too of hi
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3528
Generating text after epoch: 13
----- diversity: 0.2
----- Generating with seed: "saber" ("gay science," in ordinary langu"
saber" ("gay science," in ordinary language and desire and subjection of the strength of the standpoints and the sense of the same that the same to the standpoints of the standpoints of the standpoints of the standpoints and that the same that the science of the standpoints and that the problem of the standpoints of the sense of the state of the standpointaries of the same self-desire to the standard of the soul" of the philosophers of 
----- diversity: 0.5
----- Generating with seed: "saber" ("gay science," in ordinary langu"
saber" ("gay science," in ordinary language to the sacrifices, have have reached to stay, or to platoty and with everything of possible that who even to the fact, and in a master of man for even the plator of the standpoint of the intit believer which is the powerful
to be in the error the behaving that the stapily daren was a self-pathosop of the artists of a moral fact to the conscience the finer than a specially to the sulf to the st
----- diversity: 1.0
----- Generating with seed: "saber" ("gay science," in ordinary langu"
saber" ("gay science," in ordinary language and kind of morality, of the will who
disposition of
commontice for everything are easily?--because is not into be happening right to the
sens of authority become and deserning and have self-fact meghol! a more to still danger of
the ssochable, or, through admiration in the last reny as to find illmfelfle--too, knignic called and its "sensuse first, if the recolusing and lakes old world
are en
----- diversity: 1.2
----- Generating with seed: "saber" ("gay science," in ordinary langu"
saber" ("gay science," in ordinary langual, harss
it i old our "new geruany fto seadernt against revatuable tronopagation at insponsicive a
number of busenly are from one liberations. there is we agess?
```
</div>
    
<div class="k-default-codeblock">
```
eom,
that he who approparodice stupidity .
orightness
with learn forrigning after-ryous hearted,
and translan, anwwarking of evileffley myselves that a way yeed, the regarding
utwhound afous oftcione wors instinctions--istymly, overtain
1565/1565 [==============================] - 7s 5ms/step - loss: 1.3466
Generating text after epoch: 14
----- diversity: 0.2
----- Generating with seed: "ause in the latter case we assume a volu"
ause in the latter case we assume a voluntary to the sense and sense of the self-conscience and self-desires and and and the sense and sense--and and the hate and something the sense of the sense of the same deeper-day the and and and the strength of the strength of the strength of the strength of the same delight in the strength of the same distinction of the strength of a soul and and and the same prograte and sense, and so that the s
----- diversity: 0.5
----- Generating with seed: "ause in the latter case we assume a volu"
ause in the latter case we assume a voluntary lives of the sightent and considerable as that a state of the world of the sades to the true condition it is not men and the existence, and not in the modern science by a superiors which are interming and seems to be closed a submirit, and as the procreated for itself more religious
expecially
in the standed the master than any longest that under the destrupted in the
foreing conditional pro
----- diversity: 1.0
----- Generating with seed: "ause in the latter case we assume a volu"
ause in the latter case we assume a volu,y and
nature consequently innonce, nowles. will hence one discoants modern prosication of the spirit.
```
</div>
    
    
<div class="k-default-codeblock">
```
ie, and occise to constant by phuncentss! are
admin and fertight its diskin" come overfline, sal,
as sooatomity 
seed the proin his
(inac"s[zin grassed hious with stadls--or think, something europeing raus much mrounme, by knowledge (whie to which love in is,
as
its gently of nature. but
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2882
Generating text after epoch: 32
----- diversity: 0.2
----- Generating with seed: "nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. the"
nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. the conscience of the considerations of the sense of the sense of the sensations of the sensations of the sense of the sensality of the problem of the sensuality of the sensality, and the sensality of the sensality of the sense of the sensality is the world of the sense of the present and the sense of the result of the sensality, in the sensuality of the sensality, and the sense of the sensality and 
----- diversity: 0.5
----- Generating with seed: "nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. the"
nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. the same seems to self-extisiting and longed to that better stands in the sensumed believis and definite mans thing is sooted to the profound, and perhaps a person in the end how to anything suffer in his sight of the sense of the christianity to ourselves as something every out of a philosophers of their after the sublime to the stronge and periors are the noble soul, we have a made the other in the
----- diversity: 1.0
----- Generating with seed: "nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. the"
nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. the former to steaking the narme".
```
</div>
    
<div class="k-default-codeblock">
```
.
```
</div>
    
<div class="k-default-codeblock">
```
u 
whereptain
the serve crechisy and readity be thing may for its wrongly; and the
protible, in a thorn conceening of their inderence inkeed in the high on as a limed
in his ebuhmite
concealess for art othery that fundamentally, there is in theirs, one find to maintant and asressing considerably pesilusel, and great conceited ofo-heres as the most souls, it was 
----- diversity: 1.2
----- Generating with seed: "nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. the"
nized the woman
in such a case.
```
</div>
    
<div class="k-default-codeblock">
```
51. their to sesuntatories-programent of seen
of things, to vilithingtity, and groun canwpiceed--mannen
ansported ived
rigrrance of new mistaken with regard to the most purpose from the higherantism. still virtious jewses it with a .
hony eligitute."     wanting than to
feel gar-lon, through it is unigensive deceised theable, sovex to itfeling its folly, wherefse, believe are earsy they religion; but ala
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2872
Generating text after epoch: 33
----- diversity: 0.2
----- Generating with seed: "ucauld seem to view it) but as something"
ucauld seem to view it) but as something that is always to the world of the strength of the strength is the standard to the strength of the sense of the strength are the and instance, and the more than a stands the sense of the sense of the strength of the sense of the standine of the problem of the strength is the standard to the possibility of the standpoints is a man is a probably the sense of the conscience of the sense of the spiri
----- diversity: 0.5
----- Generating with seed: "ucauld seem to view it) but as something"
ucauld seem to view it) but as something moral and mistanded and success, and believe himself to the world person in the
strongest of the modern things which is always the problem once and unconsequent man is all the struggle world of the sense of the constant the stands one of the sense of seen the excernat egers and experience that they logicas in the new things the most subject, the christian implightering "excesse of the standind, a
----- diversity: 1.0
----- Generating with seed: "ucauld seem to view it) but as something"
ucauld seem to view it) but as something a invalued befor as approfith such as a sort of somother is
clash not believe in all means of
ingodlous victly process the conscience.--spece of time way an extest above act,
self-reward is certain philosophers away
endlive. does so let hustoratiesion, and loved sacrifice doops": he affect of the matters his hutter the uncernally, rover, not have has to go a contemptality" a series, and
perhaps n
----- diversity: 1.2
----- Generating with seed: "ucauld seem to view it) but as something"
ucauld seem to view it) but as something
much,
are deperally, he yes of rekeral, guilt estemady, is strongentetes, smile obligation
of tee; it, backing have a bestrain nome-the has it
isenouve himself: great theolaring, we love, at religious
still acciviting what goodness
of the genimson, uneffect. which
will innather for their
disciprdring his
pointion. swhe wave
invalwity, percessibly among whatever inthine insorual, and necessively
p
1565/1565 [==============================] - 7s 5ms/step - loss: 1.2851
Generating text after epoch: 34
----- diversity: 0.2
----- Generating with seed: "ion
of the healing art! through bad fema"
ion
of the healing art! through bad feman. the conscience of the sentiment of the sensible to the same as the subtle

```
</div>