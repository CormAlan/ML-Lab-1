Labb 1 i [[DD2421 Maskininlärning kursinformation|DD2421]] går ut på att lära ett [[Beslutsträd|beslutsträd]] tre begrepp: MONK-datamängderna, och sedan studera vad som händer när trädet växer sig för stort. Se [[Ordning inför labb 1 i Maskininlärning]] för läsordningen.
## MONK-datamängderna
Samtliga tre problem är binär klassificering över samma sex diskreta attribut:
$$ a_1,a_2,a_4\in\{1,2,3\},\qquad a_3,a_6\in\{1,2\},\qquad a_5\in\{1,2,3,4\} $$
Det ger $3\cdot3\cdot2\cdot3\cdot4\cdot2=432$ möjliga kombinationer. Varje kombination är positiv eller negativ enligt ett **dolt begrepp** som inlärningsalgoritmen aldrig får se:
- MONK-1: $(a_1=a_2)\vee(a_5=1)$
- MONK-2: $a_i=1$ för exakt två $i\in\{1,\dots,6\}$
- MONK-3: $(a_5=1\wedge a_4=1)\vee(a_5\ne4\wedge a_2\ne3)$, dessutom med 5 % felmärkta träningsexempel
Träningsmängderna innehåller 124, 169 respektive 122 exempel, medan testmängden alltid är alla 432 instanser.
## Assignment 0
Frågan är vilket av de tre problemen som är svårast för ett beslutsträd att lära sig. Svaret följer ur två egenskaper hos [[Beslutsträd|trädet]]: varje nod får bara testa **ett** attribut i taget, och ID3 väljer attribut **girigt** efter [[Informationsvinst|informationsvinsten]] utan att söka framåt.
### Svar: MONK-2 är svårast
Begreppet är en **symmetrisk räknefunktion** — klassen beror på *hur många* attribut som är 1, inte på vilka. Därför bär inget enskilt attribut information om klassen: $(1,1,2,2,2,2)$ är positiv medan $(1,1,1,2,2,2)$ är negativ, trots att bara $a_3$ skiljer dem åt.
Det slår direkt mot informationsvinsten. Över alla 432 instanser är 32,9 % positiva, och delar man upp datan på vilket attribut som helst ligger varje gren kvar på mellan 26 % och 36 % positiva. Alltså är $\mathrm{Gain}(S,A)\approx0$ för samtliga sex attribut, och det giriga attributvalet är blint redan vid roten — samma svårighet som hos paritetsfunktioner (XOR), där informationen sitter i *kombinationen* av attribut.
Därtill saknas kompakt trädrepresentation: på disjunktiv normalform krävs $\binom{6}{2}=15$ termer som var och en binder alla sex attributen, så trädet måste förgrena sig på nästan alla attribut längs varje väg och [[Överanpassning|överanpassar]] sina 169 träningsexempel.
### Varför de andra två är lättare
MONK-1 faller isär i test av enskilda attribut: $a_5=1$ gör i ett enda steg en hel gren ren (alla 108 sådana instanser är positiva, mot 50 % i datamängden som helhet), och $a_1=a_2$ skrivs ut genom att förgrena på $a_1$ och sedan fråga om $a_2$ i varje gren. MONK-3 har det strukturellt enklaste begreppet; svårigheten är i stället bruset, som får trädet att memorera de felmärkta exemplen — botbar [[Överanpassning|överanpassning]] som [[Beskärning av beslutsträd|beskärning]] tar bort, varför MONK-3 i praktiken når högst testnoggrannhet av de tre.
# Assignment 1
```python 
from dtree import entropy
from monkdata import monk1, monk2, monk3

print(entropy(monk1))
print(entropy(monk2))
print(entropy(monk3))
```
Prints:

| Dataset | Entropy            |
| ------- | ------------------ |
| Monk1   | 1.0                |
| Monk2   | 0.9571174282647708 |
| Monk3   | 0.9998061328047110 |
Since the target is binary, entropy depends on nothing except the class balance — $H = -p_+\log_2 p_+ - p_-\log_2 p_-$. So the numbers are really just a restatement of those counts:
- MONK-1 is exactly 62/62. Perfectly uniform → entropy hits its maximum of exactly 1.0 bit. Not a coincidence; the dataset was constructed balanced.
- MONK-3 is 60/62, a hair off balance, so it lands just barely under 1.0. The deviation is tiny because $H$ is flat near $p=0.5$ — it's a maximum, so the first derivative vanishes there and the falloff is quadratic. A 1.6-percentage-point shift in $p$ costs you only $2\cdot 10^{-4}$ bits.
- MONK-2 is the skewed one at 64/105, and it's the only one visibly below 1.0 — though even a 38/62 split only costs ~0.043 bits.
# Assignment 2
What entropy measures
$$H(X) = -\sum_i p_i \log_2 p_i$$
Read $-\log_2 p_i$ as the surprise of outcome $i$: a rare outcome ($p$ small) is very surprising, a certain one ($p=1$) carries zero surprise. Entropy is then just the expected surprise — the average over outcomes weighted by how often they occur. In bits, it's the average number of yes/no questions you need to pin down the outcome, given an optimal questioning strategy. 
#### Uniform distributions
When all $n$ outcomes are equally likely, $p_i = 1/n$ for every $i$, and the sum collapses:
$$H = -\sum_{i=1}^{n} \tfrac{1}{n}\log_2\tfrac{1}{n} = \log_2 n$$
This is the maximum entropy achievable over $n$ outcomes. Intuitively: nothing distinguishes the outcomes, so there's no way to guess better than chance and no way to encode the result in fewer than $\log_2 n$ bits on average. Every observation delivers the full $\log_2 n$ bits of information because you genuinely knew nothing beforehand.
#### Non-uniform distributions
As soon as probability mass concentrates on some outcomes, entropy drops. The high-probability outcomes contribute little surprise and occur often; the surprising outcomes are rare enough that they barely move the average. In the limit where one outcome has $p=1$, entropy is exactly 0 — the result is already known, so observing it tells you nothing.

The bound is therefore $0 \le H \le \log_2 n$, with the lower end at total certainty and the upper end at total ignorance.

| Distribution             | $H$ |                |
| ------------------------ | --- | -------------- |
| Two-headed coin, $(0,1)$ | 0   | No uncertainty |
| Fair coin, $(0.5,\;0.5)$ | 1   |                |

Two things to notice in that table:

The uniform case dominates at fixed $n$. Compare the fair die (2.585) against the loaded die with the same six faces (2.161) and the heavily loaded one (0.403). Same outcome space, strictly less uncertainty once the mass concentrates.

More outcomes doesn't automatically mean more entropy. The very loaded die has six possible faces but only 0.40 bits — less than a fair coin's 1.0 bit. It's the shape of the distribution that matters, not the size of the support. A large alphabet you can predict is easier than a small one you can't.

The shape of the binary case

For two classes, $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ traces a symmetric hump: 0 at both ends, peaking at 1.0 when $p = 0.5$. It's flat near the top (the maximum means $dH/dp = 0$ there, so deviations cost only quadratically) and steep near the edges. That's exactly the effect you saw in your own numbers: MONK-3 at $p_+ = 0.4918$ still scores 0.99981, while MONK-2 at $p_+ = 0.3787$ drops to 0.9571 — a much larger shift in $p$ buying a still-modest drop in $H$.

In ID3, entropy is the impurity measure at a node. Information gain is the drop in entropy from splitting on an attribute:
$$\text{Gain}(S, A) = H(S) - \sum_{v}\frac{|S_v|}{|S|}H(S_v)$$
A good attribute is one that carves the data into subsets that are individually far from uniform — mostly-positive or mostly-negative — because a low-entropy subset is one where you can confidently predict a label. Splitting to minimise weighted entropy is the same thing as splitting to maximise how much the answer tells you.
# Assignment 3
```python
from dtree import entropy, averageGain
from monkdata import attributes, monk1, monk2, monk3

monks = (monk1, monk2, monk3)

for name, monk in zip(("MONK 1", "MONK 2", "MONK 3"), monks):
    print(f"======== {name} ========")
    for attribute in attributes:
        print(f"{attribute.name} = {averageGain(monk, attribute)}")
```
Det här skrev ut:
```
======== MONK 1 ========
A1 = 0.07527255560831925
A2 = 0.005838429962909286
A3 = 0.00470756661729721
A4 = 0.02631169650768228
A5 = 0.28703074971578435
A6 = 0.0007578557158638421
======== MONK 2 ========
A1 = 0.0037561773775118823
A2 = 0.0024584986660830532
A3 = 0.0010561477158920196
A4 = 0.015664247292643818
A5 = 0.01727717693791797
A6 = 0.006247622236881467
======== MONK 3 ========
A1 = 0.007120868396071844
A2 = 0.29373617350838865
A3 = 0.0008311140445336207
A4 = 0.002891817288654397
A5 = 0.25591172461972755
A6 = 0.007077026074097326
```
# Assignment 4
