% BibTeX standard bibliography style `drdc-plain'
% Created by David Hally
% version 2.2 for BibTeX versions 0.99a or later
%                 LaTeX2e with class drdc-report
% 2020-01-30
%
% Changes in v2.0
%   The classification field should now contain the marker for the
%   reference entry.
% Changes in v2.1
%   A period added after the abbreviation ed.
% Changes in v2.2
%   Document type added for DRDC documents.
%   Bug fix: "In" now "Dans" for book crossref in French documents.
%   Fixed typo in DCIEM macro.
%
ENTRY
  { accessdate
    address
    author
    booktitle
    chapter
    classification
    edition
    editor
    establishment
    howpublished
    institution
    journal
    key
    month
    note
    number
    organization
    pages
    publisher
    school
    series
    title
    type
    url
    volume
    year
  }
  {}
  { label extra.label sort.label short.list }

INTEGERS { output.state before.all mid.sentence after.sentence
           after.block no.period no.space add.comma}

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
  #4 'no.period :=
  #5 'no.space :=
  #6 'add.comma :=
}

STRINGS { s t w }

FUNCTION{output.nonnull}
{ 's :=
  output.state mid.sentence =
    { ", " * write$ }
    { output.state no.period =
        { " " * write$ }
        { output.state no.space =
	    { "" * write$ }
	    { output.state after.block =
	        { add.period$ write$ 
	          newline$
                  "\newblock " write$
	        }
	        { output.state before.all =
		    'write$
                    { output.state add.comma = 
                        { ", " * write$ }
		        { add.period$ " " * write$ }
                      if$
                    }
	          if$
	        }
	      if$
	    }
	  if$
	}
      if$
    }
  if$
  mid.sentence 'output.state :=
  s
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem[" write$
  label write$
  ")" make.full.names duplicate$ short.list =
     { pop$ }
     { * }
   if$
  "]{" * write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}
{ add.period$ write$ 
  newline$
}

FUNCTION {new.block}
{ output.state before.all =
    'skip$
    { after.block 'output.state := }
  if$
}

FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\em " swap$ * "}" * }
  if$
}

FUNCTION {typewriter}
{ duplicate$ empty$
    { pop$ "" }
    { "{\tt " swap$ * "}" * }
  if$
}

FUNCTION {embolden}
{ duplicate$ empty$
    { pop$ "" }
    { "{\bf " swap$ * "}" * }
  if$
}

FUNCTION {inbrackets}
{ duplicate$ empty$
    { pop$ "" }
    { "(" swap$ * ")" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

FUNCTION {format.names}
{ 's :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr "{vv~}{ll}{,~f.}{,~jj}" format.name$ 't :=
      nameptr #1 >
	{ namesleft #1 >
	    { ", " * t * }
	    { numnames #2 >
		{ "," * }
		'skip$
	      if$
	      t "others" =
		{ " et~al." * }
		{ " \and\ " * t * }
	      if$
	    }
	  if$
	}
	't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

FUNCTION {format.editors}
{ editor empty$
    { "" }
    { editor format.names 
      numnames #1 >
        {", (Eds.)" * }
        {", (Ed.)" * }
      if$
    }
  if$
}

FUNCTION {format.title}
{ title empty$
    { "" }
    { title }
  if$
}

FUNCTION {format.establishment}
{ establishment empty$
    { "empty establishment in " cite$ * ": using \establishmentname" *
      warning$
      "\establishmentname"
    }
    { establishment }
  if$
}

FUNCTION {format.full.names}
{'s :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { s nameptr
      "{vv~}{ll}" format.name$ 't :=
      nameptr #1 >
        {
          namesleft #1 >
            { ", " * t * }
            {
              numnames #2 >
                { "," * }
                'skip$
              if$
              t "others" =
                { " et~al." * }
                { " \and\ " * t * }
              if$
            }
          if$
        }
        't
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {author.editor.full}
{ author empty$
    { editor empty$
        { "" }
        { editor format.full.names }
      if$
    }
    { author format.full.names }
  if$
}

FUNCTION {author.full}
{ author empty$
    { "" }
    { author format.full.names }
  if$
}

FUNCTION {editor.full}
{ editor empty$
    { "" }
    { editor format.full.names }
  if$
}

FUNCTION {make.full.names}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.full
    { type$ "proceedings" =
        'editor.full
        'author.full
      if$
    }
  if$
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
	{ t #1 #2 substring$ "--" = not
	    { "--" *
	      t #2 global.max$ substring$ 't :=
	    }
	    {   { t #1 #1 substring$ "-" = }
		{ "-" *
		  t #2 global.max$ substring$ 't :=
		}
	      while$
	    }
	  if$
	}
	{ t #1 #1 substring$ *
	  t #2 global.max$ substring$ 't :=
	}
      if$
    }
  while$
}

FUNCTION {format.date}
{ year empty$
    { month empty$
	{ "" }
	{ "there's a month but no year in " cite$ * warning$
	  month inbrackets no.period 'output.state :=
	}
      if$
    }
    { month empty$
	'year
	{ month " " * year * }
      if$
      inbrackets no.period 'output.state :=
    }
  if$
}

FUNCTION{format.adate}
{ year empty$
    { "" }
    { year inbrackets no.period 'output.state := }
  if$
}

FUNCTION{format.accessdate}
{ accessdate empty$
    { "" }
    { "\accessdate: " accessdate * inbrackets no.period 'output.state := }
  if$
}

FUNCTION {format.btitle}
{ title
}

FUNCTION {tie.or.space.connect}
{ % Always use a tie; this looks best since we use raggedright 
  % bibliography [DH: 2005-06-03]
  "~" swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

FUNCTION{format.ttitle}
{ title
}

FUNCTION {format.bvolume}
{ volume empty$
    { "" }
    { "Vol." volume tie.or.space.connect
      series empty$
	'skip$
	{ " \of\ " * series emphasize * }
      if$
      "volume and number" number either.or.check
    }
  if$
}

FUNCTION {format.number.series}
{ volume empty$
    { number empty$
	{ series field.or.null }
	{ "Number" number tie.or.space.connect
	  series empty$
	    { "there's a number but no series in " cite$ * warning$ }
	    { " \in\ " * series * }
	  if$
	}
      if$
    }
    { "" }
  if$
}

FUNCTION {format.drdc.number}
{
  type empty$
    { "" }
    { type " " * }
  if$
  number empty$
    { pop$ }
    { number * inbrackets }
  if$
}

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { edition "t" change.case$ " ed." * }
  if$
}

INTEGERS { multiresult }

FUNCTION {multi.page.check}
{ 't :=
  #0 'multiresult :=
    { multiresult not
      t empty$ not
      and
    }
    { t #1 #1 substring$
      duplicate$ "-" =
      swap$ duplicate$ "," =
      swap$ "+" =
      or or
	{ #1 'multiresult := }
	{ t #2 global.max$ substring$ 't := }
      if$
    }
  while$
  multiresult
}

FUNCTION {format.pages}
{ pages empty$
    { "" }
    { pages multi.page.check
	{ "pp." pages n.dashify tie.or.space.connect }
	{ "p." pages tie.or.space.connect }
      if$
    }
  if$
}

FUNCTION {format.vol.num.pages}
{ 
  number empty$
    { pages empty$
        { volume empty$
            { "" }
            { "Vol." volume tie.or.space.connect }
          if$
        }
        { volume field.or.null }
      if$
    }
    { volume field.or.null
      "(" number * ")" * *
      volume empty$
	{ "there's a number but no volume in " cite$ * warning$ }
	'skip$
      if$
    }
  if$
  pages empty$
    'skip$
    { duplicate$ empty$
	{ pop$ format.pages }
	{ ", " * pages n.dashify * }
      if$
    }
  if$
}

FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
	{ "Ch." }
	{ type "t" change.case$ }
      if$
      chapter tie.or.space.connect
      pages empty$
	'skip$
	{ ", " * format.pages * }
      if$
    }
  if$
}

FUNCTION {format.in.ed.booktitle}
{ booktitle empty$
    { "" }
    { editor empty$
	{ "\In\ " booktitle emphasize * }
	{ "\In\ " format.editors * ", " * booktitle emphasize * }
      if$
    }
  if$
}

FUNCTION {empty.misc.check}
{ author empty$ title empty$ howpublished empty$
  month empty$ year empty$ note empty$
  and and and and and
  key empty$ not and
    { "all relevant fields are empty in " cite$ * warning$ }
    'skip$
  if$
}

FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type "t" change.case$
    }
  if$
}

FUNCTION {format.tr.number}
{ number empty$
    'skip$
    { number tie.or.space.connect inbrackets }
  if$
}

FUNCTION {format.article.crossref}
{ key empty$
    { journal empty$
	{ "need key or journal for " cite$ * " to crossref " * crossref *
	  warning$
	  ""
	}
	{ "\In\ {\em " journal * "\/}" * }
      if$
    }
    { "\In\ " key * }
  if$
  " \cite[]{" * crossref * "}" *
}

FUNCTION {format.crossref.editor}
{ editor #1 "{vv~}{ll}" format.name$
  editor num.names$ duplicate$
  #2 >
    { pop$ " et~al." * }
    { #2 <
	'skip$
	{ editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
	    { " et~al." * }
	    { " \and\ " * editor #2 "{vv~}{ll}" format.name$ * }
	  if$
	}
      if$
    }
  if$
}

FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      "\In\ "
    }
    { "Vol." volume tie.or.space.connect
      " \of\ " *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ series empty$
	    { "need editor, key, or series for " cite$ * " to crossref " *
	      crossref * warning$
	      "" *
	    }
	    { "{\em " * series * "\/}" * }
	  if$
	}
	{ key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite[]{" * crossref * "}" *
}

FUNCTION {format.incoll.inproc.crossref}
{ editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ booktitle empty$
	    { "need editor, key, or booktitle for " cite$ * " to crossref " *
	      crossref * warning$
	      ""
	    }
	    { "\In\ {\em " booktitle * "\/}" * }
	  if$
	}
	{ "\In\ " key * }
      if$
    }
    { "\In\ " format.crossref.editor * }
  if$
  " \cite[]{" * crossref * "}" *
}

FUNCTION {format.address.publisher}
{
  address empty$
    { publisher }
    { address
      publisher empty$
        'skip$
        {  ": " * publisher * }
      if$
    }
  if$
}

FUNCTION{classification.output}
{ classification empty$
    { "\U" output }
    { classification output }
  if$
  no.period 'output.state :=
}

FUNCTION{format.url}
{ url empty$
    { "" }
    { "\url{" url * "}" * }
  if$
}

FUNCTION {article}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.title "title" output.check
  crossref missing$
    { journal emphasize "journal" output.check
      format.vol.num.pages add.comma 'output.state := output
    }
    { format.article.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {book}
{ output.bibitem
  classification.output
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull
      crossref missing$
	{ "author and editor" editor either.or.check }
	'skip$
      if$
    }
  if$
  format.adate "year" output.check
  format.btitle "title" output.check
  format.edition add.comma 'output.state := output
  crossref missing$
    { format.bvolume output
      format.number.series output
      format.address.publisher "publisher" output.check
    }
    { format.book.crossref output.nonnull }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {booklet}
{ output.bibitem
  classification.output
  format.authors output
  format.adate output
  format.title "title" output.check
  new.block
  howpublished output
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {inbook}
{ output.bibitem
  classification.output
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull
      crossref missing$
	{ "author and editor" editor either.or.check }
	'skip$
      if$
    }
  if$
  format.adate "year" output.check
  format.btitle "title" output.check
  format.edition add.comma 'output.state := output
  crossref missing$
    { format.bvolume add.comma 'output.state := output
      format.number.series add.comma 'output.state := output
      format.chapter.pages "chapter and pages" 
      add.comma 'output.state := output.check
      format.address.publisher "publisher" output.check
    }
    { format.chapter.pages "chapter and pages" output.check
      format.book.crossref output.nonnull
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {incollection}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.title "title" output.check
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.edition add.comma 'output.state := output
      format.bvolume add.comma 'output.state := output
      format.number.series add.comma 'output.state := output
      format.chapter.pages add.comma 'output.state := output
      format.address.publisher "publisher" output.check
    }
    { format.incoll.inproc.crossref output.nonnull
      format.chapter.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {inproceedings}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.title "title" output.check
  crossref missing$
    { format.in.ed.booktitle "booktitle" output.check
      format.bvolume add.comma 'output.state := output
      format.number.series add.comma 'output.state := output
      format.pages add.comma 'output.state := output
      organization output
      format.address.publisher output
    }
    { format.incoll.inproc.crossref output.nonnull
      format.pages output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {conference} { inproceedings }

FUNCTION {manual}
{ output.bibitem
  classification.output
  format.authors output
  format.adate output
  format.btitle "title" output.check
  format.edition output
  organization empty$ address empty$ and year empty$ and
    'skip$
    { organization output
      address output
    }
  if$
  new.block
  note output
  fin.entry
}

FUNCTION {mastersthesis}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.ttitle "title" output.check
  "\masters" format.thesis.type output.nonnull
  school "school" output.check
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {misc}
{ output.bibitem
  classification.output
  format.authors output
  format.adate output
  format.title output
  new.block
  howpublished output
  new.block
  note output
  fin.entry
  empty.misc.check
}

FUNCTION {phdthesis}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.ttitle "title" output.check
  "\phd" format.thesis.type output.nonnull
  school "school" output.check
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {proceedings}
{ output.bibitem
  classification.output
  editor empty$
    { organization output }
    { format.editors output.nonnull }
  if$
  format.adate "year" output.check
  format.btitle "title" output.check
  format.bvolume add.comma 'output.state := output
  format.number.series add.comma 'output.state := output
  organization output
  format.address.publisher output
  new.block
  note output
  fin.entry
}

FUNCTION {techreport}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.ttitle "title" output.check
  type empty$
    { "Technical Report" }
    'type
  if$
  format.tr.number output.nonnull no.period 'output.state :=
  institution "institution" output.check
  address output
  new.block
  note output
  fin.entry
}

FUNCTION {unpublished}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate output
  format.title "title" output.check
  new.block
  note "note" output.check
  fin.entry
}

FUNCTION {drdc.report}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.ttitle "title" output.check
  format.drdc.number "number" output.check no.period 'output.state :=
  format.pages output
  format.establishment output
  new.block
  note output
  fin.entry
}

function{ttcpnato}
{ output.bibitem
  classification.output
  format.authors output
  format.adate output
  format.ttitle "title" output.check
  organization output
  address output
  new.block
  note output
  fin.entry
}

function{oral}
{ output.bibitem
  classification.output
  format.authors "author" output.check
  format.adate "year" output.check
  format.ttitle "title" output.check
  institution "institution" output.check
  address output
  new.block
  note output
  fin.entry
}

function{web}
{ output.bibitem
  classification.output
  format.authors output
  format.date output
  format.title "title" output.check
  "(\online)" no.period 'output.state := output
  format.address.publisher output
  format.url "url" output.check
  format.accessdate "accessdate" output.check
  new.block
  note output
  fin.entry
}

FUNCTION{seminar} {oral}

FUNCTION{briefing} {oral}

FUNCTION{irc} { drdc.report}

FUNCTION {default.type} { misc }

MACRO {jan} {"\numtomonth{1}"}
MACRO {feb} {"\numtomonth{2}"}
MACRO {mar} {"\numtomonth{3}"}
MACRO {apr} {"\numtomonth{4}"}
MACRO {may} {"\numtomonth{5}"}
MACRO {jun} {"\numtomonth{6}"}
MACRO {jul} {"\numtomonth{7}"}
MACRO {aug} {"\numtomonth{8}"}
MACRO {sep} {"\numtomonth{9}"}
MACRO {oct} {"\numtomonth{10}"}
MACRO {nov} {"\numtomonth{11}"}
MACRO {dec} {"\numtomonth{12}"}

MACRO {DRDC} {"Defence Research and Development Canada"}
MACRO {RDDC} {"Recherche et d\'eveloppement pour la d\'efense Canada"}

% The following macros have been retained so .bib files will not
% break.  They should not be used for new documents.
MACRO {DCIEM} {"Defence and Civil Institute of Environmental Medecine"}
MACRO {DREA}  {"Defence Research Establishment Atlantic"}
MACRO {DREO}  {"Defence Research Establishment Ottawa"}
MACRO {DRES}  {"Defence Research Establishment Suffield"}
MACRO {DREV}  {"Defence Research Establishment Valcartier"}

MACRO {DRDCA}  {"Defence Research and Development Canada -- Atlantic Research Centre"}
MACRO {DRDCO}  {"Defence Research and Development Canada -- Ottawa Research Centre"}
MACRO {DRDCS}  {"Defence Research and Development Canada -- Suffield Research Centre"}
MACRO {DRDCT}  {"Defence Research and Development Canada -- Toronto Research Centre"}
MACRO {DRDCV}  {"Defence Research and Development Canada -- Valcartier Research Centre"}
MACRO {DRDCCORA}   {"Defence Research and Development Canada -- Centre for Operational Research and Analysis"}
MACRO {DRDCCSS}    {"Defence Research and Development Canada -- Centre for Security Science"} 
MACRO {DRDCDGMPRA} {"Defence Research and Development Canada -- Director General Military Personnel Research and Analysis"} 

MACRO {IMED} {"Institut de m\'edecine environnementale pour la d\'efense"}
MACRO {CRDA} {"Centre de recherches pour la d\'efense, Atlantique"}
MACRO {CRDO} {"Centre de recherches pour la d\'efense, Ottawa"}
MACRO {CRDS} {"Centre de recherches pour la d\'efense, Suffield"}
MACRO {CRDV} {"Centre de recherches pour la d\'efense, Valcartier"}

MACRO {RDDCA} {"Recherche et d\'eveloppement pour la d\'efense Canada -- Centre de recherches de l'Atlantique"} 
MACRO {RDDCO} {"Recherche et d\'eveloppement pour la d\'efense Canada -- Centre de recherches d'Ottawa"}
MACRO {RDDCS} {"Recherche et d\'eveloppement pour la d\'efense Canada -- Centre de recherches de Suffield"}
MACRO {RDDCT} {"Recherche et d\'eveloppement pour la d\'efense Canada -- Centre de recherches de Toronto"}
MACRO {RDDCV} {"Recherche et d\'eveloppement pour la d\'efense Canada -- Centre de recherches de Valcartier"}
MACRO {RDDCCARO}   {"Recherche et d\'eveloppement pour la d\'efense Canada -- Centre d'analyse et de recherche op\'erationelle"} 
MACRO {RDDCCSS}    {"Recherche et d\'eveloppement pour la d\'efense Canada -- Centre des sciences pour la s\'ecurit\'e"} 
MACRO {RDDCDGRAPM} {"Recherche et d\'eveloppement pour la d\'efense Canada -- Directeur g\'en\'eral -- Recherche et analyse (Personnel militaire)"}

READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

INTEGERS { len }

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}

FUNCTION {format.lab.names}
{ 's :=
  s #1 "{vv~}{ll}" format.name$
  s num.names$ duplicate$
  #2 >
    { pop$ " et~al." * }
    { #2 <
        'skip$
        { s #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
            { " et~al." * }
            { " \and\ " * s #2 "{vv~}{ll}" format.name$ * }
          if$
        }
      if$
    }
  if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
        { cite$ #1 #3 substring$ }
        'key
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
        { key empty$
            { cite$ #1 #3 substring$ }
            'key
          if$
        }
        { editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.key.organization.label}
{ author empty$
    { key empty$
        { organization empty$
            { cite$ #1 #3 substring$ }
            { "The " #4 organization chop.word #3 text.prefix$ }
          if$
        }
        'key
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.organization.label}
{ editor empty$
    { key empty$
        { organization empty$
            { cite$ #1 #3 substring$ }
            { "The " #4 organization chop.word #3 text.prefix$ }
          if$
        }
        'key
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.short.authors}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
        'editor.key.organization.label
        { type$ "manual" =
            'author.key.organization.label
            'author.key.label
          if$
        }
      if$
    }
  if$
  'short.list :=
}

FUNCTION {calc.label}
{ calc.short.authors
  short.list
  "("
  *
  year duplicate$ empty$
  short.list key field.or.null = or
     { pop$ "" }
     'skip$
  if$
  *
  'label :=
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
        { "   " * }
        'skip$
      if$
      s nameptr "{vv{ } }{ll{ }}{  ff{ }}{  jj{ }}" format.name$ 't :=
      nameptr numnames = t "others" = and
        { "et al" * }
        { t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "The " #4 t chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
        { "to sort, need author or key in " cite$ * warning$
          ""
        }
        { key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
        { key empty$
            { "to sort, need author, editor, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
        { key empty$
            { "to sort, need author, organization, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
        { key empty$
            { "to sort, need editor, organization, or key in " cite$ * warning$
              ""
            }
            { key sortify }
          if$
        }
        { "The " #4 organization chop.word sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

FUNCTION {presort}
{ calc.label
  label sortify
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  or
    'author.editor.sort
    { type$ "proceedings" =
        'editor.organization.sort
        { type$ "manual" =
            'author.organization.sort
            'author.sort
          if$
        }
      if$
    }
  if$
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.label :=
  sort.label *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}

SORT

STRINGS { longest.label last.label next.extra }

INTEGERS { longest.label.width last.extra.num number.label }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #0 int.to.chr$ 'last.label :=
  "" 'next.extra :=
  #0 'longest.label.width :=
  #0 'last.extra.num :=
  #0 'number.label :=
}

FUNCTION {forward.pass}
{ last.label label =
    { last.extra.num #1 + 'last.extra.num :=
      last.extra.num int.to.chr$ 'extra.label :=
    }
    { "a" chr.to.int$ 'last.extra.num :=
      "" 'extra.label :=
      label 'last.label :=
    }
  if$
  number.label #1 + 'number.label :=
}

FUNCTION {reverse.pass}
{ next.extra "b" =
    { "a" 'extra.label := }
    'skip$
  if$
  extra.label 'next.extra :=
  extra.label
  duplicate$ empty$
    'skip$
    { "{\natexlab{" swap$ * "}}" * }
  if$
  'extra.label :=
  label extra.label * 'label :=
}

EXECUTE {initialize.longest.label}

ITERATE {forward.pass}

REVERSE {reverse.pass}

FUNCTION {bib.sort.order}
{ sort.label  'sort.key$ :=
}

ITERATE {bib.sort.order}

SORT

FUNCTION {begin.bib}
{   preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{" number.label int.to.str$ * "}" *
  write$ newline$
  "\makeatletter" write$ newline$
  "\@ifundefined{natexlab}{\def\natexlab#1{#1}}{}" write$ newline$
  "\@ifundefined{url}{" write$ newline$
  "  \def\url{\begingroup\catcode`\~=12\catcode`\_=12%" write$ newline$
  "           \deactivatepunctuation\DRDC@url}" write$ newline$
  "  \def\DRDC@url#1{\texttt{#1}\endgroup}}{}" write$ newline$
  "\ifDRDC@isfrench" write$ newline$
  "  \def\in{dans}\def\In{Dans}\def\of{de}\def\and{et}" write$ newline$
  "  \def\online{en~ligne}\def\accessdate{Date d'acc\`es}" write$ newline$
  "  \def\masters{th\`ese de Ma\^itrise}\def\phd{th\`ese de Doctorat}"
  write$ newline$
  "\else" write$ newline$
  "  \def\in{in}\def\In{In}\def\of{of}\def\and{and}" write$ newline$
  "  \def\online{online}\def\accessdate{Access Date}" write$ newline$
  "  \def\masters{Master's thesis}\def\phd{Ph.D.\ thesis}" write$ newline$
  "\fi" write$ newline$
  "\ifDRDC@isclassified\else\def\U{}\fi" write$ newline$
  "\makeatother" write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}
