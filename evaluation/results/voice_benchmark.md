# Voice Backend Benchmark

## superb

| File | True | Pred | Conf |
|------|------|------|------|
| Happy2.wav | happy | happy | 0.918 |
| happy_sample.wav | happy | sad | 0.906 |
| angry_sample.wav | angry | sad | 0.972 |
| Neutral1.wav | neutral | angry | 0.517 |
| Neutral2.wav | neutral | angry | 0.884 |
| Sad1.wav | sad | happy | 0.670 |
| Sad2.wav | sad | angry | 0.433 |
| Sad3.wav | sad | happy | 0.490 |

**Accuracy:** 12.5%  
**Macro-F1:** 0.100 (4 classes; other=wrong)  
**Mean conf correct:** 0.918  
**Mean conf wrong:** 0.696  
**Mean s/clip:** 1.226  

**Confusion matrix** (rows=true, cols=pred):

```
        neut    happ     sad    angr
neutra     0       0       0       2
happy      0       1       1       0
sad        0       2       0       1
angry      0       0       1       0
```

## emotion2vec

| File | True | Pred | Conf |
|------|------|------|------|
| Happy2.wav | happy | happy | 1.000 |
| happy_sample.wav | happy | sad | 0.998 |
| angry_sample.wav | angry | neutral | 1.000 |
| Neutral1.wav | neutral | sad | 1.000 |
| Neutral2.wav | neutral | neutral | 1.000 |
| Sad1.wav | sad | sad | 0.993 |
| Sad2.wav | sad | sad | 0.999 |
| Sad3.wav | sad | sad | 1.000 |

**Accuracy:** 62.5%  
**Macro-F1:** 0.479 (4 classes; other=wrong)  
**Mean conf correct:** 0.998  
**Mean conf wrong:** 0.999  
**Mean s/clip:** 3.170  

**Confusion matrix** (rows=true, cols=pred):

```
        neut    happ     sad    angr
neutra     1       0       1       0
happy      0       1       1       0
sad        0       0       3       0
angry      1       0       0       0
```

