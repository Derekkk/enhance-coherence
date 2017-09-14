DUC 2006 PYRAMID DATA AND SCORING SCRIPT
================================================
Table of Contents of DUC2006pyramiddata.tar.gz:
================================================
1. Annotated peer files (allpans/*.pan)

2. Pyramids (allpyramids/*.pyr)

3. DUCView (scoring/DUCView-1.4.jar)

4. Other data for the score table (scoring/*.table, scoring/2006_pyramid_creators.txt scoring/2006-data-description.txt)

5. Scoring script (scoring/create-06-table.txt)

6. Pyramid score table and header file (scoring/2006_modified_scores.txt, scoring/2006-header.txt)

================================================
Details:
================================================
1. Annotated peer files (allpans/*.pan)

The annotated peer files are in the allpans directory created by
unpacking the *tar.gz file.  Twenty-one annotator groups/sites
participated in annotating peer summaries for twenty document
sets. One document set (D0631) was annotated by two sites
independently. There were twenty-two peer summaries per document set
that were annotated, one of which is a baseline summary. In sum, there
are 462 peer annotations (22 files for each of 19 document sets, and
44 for another document set that was annotated twice).  These are in
XML files with the extension "pan" -- produced by using the DUCView
annotation tool. The first field of each *pan file name is the
annotator id, and the rest of the file name is identical to the
original peer file name.

2. Pyramids (allpyramids/*.pyr)

Four sites participated in creating twenty pyramids for the DUC 2006
pyramid effort: Columbia University, Mitre, Microsoft, and National
University of Singapore (NUS). Each pyramid annotator was paired with
a partner who reviewed the first pass results. The final pyramids were
arrived at through adjudication with the partner sites. A special
thanks go to Mitre and the NUS team, who helped create pyramids even
though they were not participating in the remaining parts of the
evaluation.

3. DUCView (scoring/DUCView-1.4.jar)

The annotation tool, DUCView, is a cross-platform annotation tool
written in java that facilitates creating the pyramids and scoring
peer summaries in an intuitive graphical environment. To create a
pyramid, the annotator starts with a plaintext file containing the
model summaries and selects pieces of text, adding them to the a
graphical tree representation as new SCUs or SCU contributors.  Peer
annotation is done in a similar environment; the annotator starts with
an existing pyramid, and loads plaintext files containing the peer
summaries. DUCView features, and its use for pyramid and peer
annotation, are described on the webpages for the 2006 pyramid DUC
effort:
http://www1.cs.columbia.edu/~becky/DUC2006/2006-pyramid-guidelines.html#start.

4. Other data for the score table (scoring/*.table, scoring/2006_pyramid_creators.txt, scoring/2006-data-description.txt)

Three files provide additional information for the scoring script and
score table: 2006_linguistic_quality.table, 2006_pyramid_creators.txt
and 2006_content.table. For more information, see the
"2006-data-description.txt" file.

5. Scoring script (scoring/create-06-table.txt)

The perl script "create-06-table.pl" takes a directory argument: a
directory with all the original *.pan files.  To run it, the following
files should be in the same directory that the script is run from:
DUCView-1.4.jar, 2006_linguistic_quality.table,
2006_pyramid_creators.txt and 2006_content.table.

6. Pyramid score table and header file (scoring/2006_modified_scores.txt, scoring/header.txt)

The modified pyramid scores for the peer annotation files are in a
table in the file "2006_modified_scores.txt". DUCView-1.4.jar can
output two types of pyramid scores, original and modified, but for
this year's DUC we use only the modified score. There are thirteen
fields in the table, in this order: "setid", "peerid",
"modifiedscore", "numSCUs", "numrepetitions", "annotatorid",
"pyramidcreator", 5 fields for linguistic quality ("lingqual1" through
"lingqual5"), and finally, "content". In sum, the table dimensions are
462 rows (one per pan file) by 13 columns. A file containing only the
column headers is included: 2006-header.txt. More information on the
column contents is in the "2006-data-description.txt" file.


