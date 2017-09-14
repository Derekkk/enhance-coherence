
		DUC 2007 Update Task: Evaluation Results
		========================================


This is the README file for the manual and automatic evaluation that
was performed at NIST for summaries in the DUC 2007 update task.

Ten NIST assessors wrote summaries for the 10 topics in the DUC 2007
update task.  The documents for each topic were divided into 3
subsets, A-C, and 4 human summaries were written for each subset.  The
human summarizer IDs are A-J.

Two baseline summarizers were included in the evaluation:

    Baseline 1 (summarizer ID = 35): return all the leading sentences
    (up to 100 words) in the <TEXT> field of the most recent document.

    Baseline 2 (summarizer ID = 58): CLASSY04, a summarizer that
    ignores the topic narrative but that had the highest mean SEE
    coverage score in Task 2 of DUC 2004, a multi-document
    summarization task.  The system uses the CLASSY04 HMM described in
    <http://duc.nist.gov/pubs/2004papers/ida.conroy.ps> with signature
    terms as observables and the pivoted QR method for redundancy
    removal.  The sentences are chosen only from the most recent
    collection of documents.  For example, the summary for D0703A-B
    selects sentences only from the 8 articles in this cluster;
    however, it uses D0703A-A in the computation of signature terms.
    Likewise, the summary for D0703A-C selects sentences from only the
    7 documents in this cluster and only uses D0703A-A and D0703A-B in
    the computation of signature terms.

NIST received submissions from 22 different participants for the
update task.  The participants' summarizer IDs are 36-57.

All summaries were truncated to 100 words before being evaluated.
Summaries are saved one per file, using the following naming
convention: <topic>-<subset>.M.100.<topic_selectorID>.<summarizerID>



Responsiveness
--------------

Responsiveness: NIST assessors assigned a content responsiveness score
to each of the automatic and human summaries.  The content score is an
integer between 1 (very poor) and 5 (very good) and is based on the
amount of information in the summary that helps to satisfy the
information need expressed in the topic narrative.  

The assessor for a given topic had previously read all the documents
and written a summary for each of the A, B, and C subsets.  As a
surrogate for rereading all the documents at assessment time, the
assessor was given the 4 human summaries for each subset.  When
evaluating the update summaries for a particular subset of documents,
the assessor was reminded that the intended user had already read
documents in the earlier subsets.  Therefore, information in a summary
for subset B that was already in subset A should be discounted;
similarly, information in a summary for subset C that was already in
subsets A and B should be discounted.

Files under Responsiveness/:
   content.table		scores for content responsiveness
   avg_content.all		average of all content scores, by summarizer ID
   avg_content.A		average of content scores for subset A, by summarizer ID
   avg_content.B		average of content scores for subset B, by summarizer ID
   avg_content.C		average of content scores for subset C, by summarizer ID



Pyramid
-------

Nine NIST assessors created pyramids and annotated the peer summaries
for the update task using the guidelines described in:
      http://www1.cs.columbia.edu/~becky/DUC2006/2006-pyramid-guidelines.html

For each of the 10 update topics, a single assessor created all the
pyramids and annotated all the peer summaries.  NIST staff checked the
pyramids of Assessors A, G, and H, but no other checking of pyramids
or peer annotations was done. Assessors A, G, and H expressed the
greatest difficulty in understanding and implementing the pyramid
evaluation method; they also used a more qualitative method for
assessing content responsiveness, while other assessors devised their
own (various) quantitative methods for assigning content
responsiveness scores.

Files under Pyramid/:
   allpyramids/*.pyr:			pyramids
   allpans/*.pan:			annotated peer files
   scoring/:
	create-07-table.pl		scoring script
	2007_modified_scores.txt	score table
	2007-header.txt			header file for the score table
	2007-data-description.txt	description of each field in the score table
	DUCView-1.4.jar			DUCView (used by scoring script)
	2007_content.table		content responsiveness scores (used by scoring script)
	2007_pyramid_creators.txt	assessor who created each pyramid (used by scoring script)



ROUGE
-----

ROUGE-2 and ROUGE-SU4 scores were computed by running ROUGE-1.5.5 with
stemming but no removal of stopwords.  The input file implemented
jackknifing so that scores of systems and humans could be
compared. 

ROUGE run-time parameters:
	ROUGE-1.5.5.pl -n 4 -w 1.2 -m  -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -d rougejk.in 

Files under ROUGE/:
   models/		sentence-segmented human summaries
   peers/		sentence-segmented human and automatic summaries
   rougejk.in		input file to ROUGE-1.5.5
   rougejk.m.out	output of ROUGE-1.5.5
   rouge2.jk.m.avg	average ROUGE-2 recall, by summarizer ID
   rougeSU4.jk.m.avg	average ROUGE-SU4 recall, by summarizer ID



Basic Elements
--------------

Basic Elements (BE) scores were computed by first using the tools in
BE-1.1 to extract BE-F from each sentence-segmented <summary>:
   bebreakMinipar.pl -p $MINIPATH <summary>

The BE-F were then matched by running ROUGE-1.5.5 with stemming, using
the Head-Modifier (HM) matching criterion.  The input file to
ROUGE-1.5.5 implemented jackknifing so that scores of systems and
humans could be compared.  

BE run-time parameters:
	ROUGE-1.5.5.pl -m -a -d -3 HM simplejk.in  

Files under BE/:
   BEmodels/			BEs from human summaries
   BEpeers/			BEs from human and automatic summaries
   simplejk.in			input file to ROUGE-1.5.5
   simplejk.m.hm.out		output of ROUGE-1.5.5
   simple.jk.m.hm.avg		average BE recall, by summarizer ID



