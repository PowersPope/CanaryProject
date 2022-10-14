# Modeling the statistical structure of a complex birdsong

This project is in conjunction with Tim Gardner at the University of Oregon.

Project worked on by Andrew Powers (UO), Rachel Thompson (OSU), and Noel Lefevre (OSU)

Last Update October 2022

## Project Description

Bird songs range in form from the simple notes of a Chipping Sparrow to the complex repertoire of 
the nightingale. Recent studies suggest that bird songs may contain non-adjacent dependencies where 
the choice of what to sing next depends on the history of what has already been produced. However, 
the complexity of these rules has not been examined statistically. This project will examine one 
complex singer—the domesticated canary— and determine if their songs are influenced by long-range 
rules. The choice of how long to repeat a given note or which note to choose next may depend on the 
history of the song. Like most forms of human music, do the songs of canaries contain patterns 
expressed over long timescales? Are their songs governed by rules that apply to multiple levels of 
a temporal hierarchy? We will develop time-series analysis methods such as Markov chain models 
or Hidden Markov models to build a statistical model of sequences of canary song.


## Progress

Currently we have a working python script that can calculate the One Step Markov Model probabilites
of which phrase follows another. This is a transition matrix. This script was built with
made up test data and still needs to be confirmed on the real data.

Given a dataframe (example):

```
string,duration
ABJR,140
NFRK,40
ABIE,50
```

We can calculate the transition matrix. With the matrix being *MxM* 
M = Total Number of Song Characters

```
$ python transition_matrix.py
[[0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.33333333 0.         0.
  0.         0.         0.         0.         0.66666667]
 [1.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         1.
  0.         0.         0.         0.         0.        ]
 [0.         0.         1.         0.         0.         0.
  0.         0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         1.         0.        ]
 [0.         0.         0.         0.         0.5        0.
  0.         0.5        0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  0.         0.         0.         1.         0.        ]
 [1.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.        ]
 [0.5        0.         0.         0.         0.         0.
  0.         0.         0.5        0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  1.         0.         0.         0.         0.        ]]
```
