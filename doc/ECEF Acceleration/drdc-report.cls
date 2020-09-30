%%
%% This is file `drdc-report.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% drdc-classes.dtx  (with options: `report')
%% 
%% \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{drdc-report}
              [2017/12/18 v3.0
 LaTeX class for DRDC reports]
\def\[{%
   \relax\ifmmode
      \@badmath
   \else
      \ifvmode
         \nointerlineskip
         \makebox[.6\linewidth]{}% standard version leaves out {}
      \fi
      $$%%$$ BRACE MATCH HACK
   \fi
}
\newcommand\DRDC@error[1]{\ClassError{drdc-report}{#1}{}}
\newif\ifDRDC@switchnotdone
\newenvironment{DRDC@switch}[1]{%
  \DRDC@switchnotdonetrue
  \edef\DRDC@swtemp{#1}%
  \def\case##1##2{%
    \ifDRDC@switchnotdone
      \edef\DRDC@swtempa{##1}%
      \ifx\DRDC@swtemp\DRDC@swtempa ##2\DRDC@switchnotdonefalse\fi
    \fi\ignorespaces}%
  \def\default##1{\ifDRDC@switchnotdone ##1\fi\ignorespaces}%
  \ignorespaces
}{}
\@twocolumnfalse
\newif\ifDRDC@ishyperref
\AtBeginDocument{%
  \@ifpackageloaded{hyperref}{\DRDC@ishyperreftrue}%
                             {\DRDC@ishyperreffalse}}
\newif\ifDRDC@isclassified
\AtBeginDocument{%
  \@ifpackageloaded{drdc-classified}{\DRDC@isclassifiedtrue}%
                                    {\DRDC@isclassifiedfalse}}
\newlength{\DRDC@templen}
\newlength{\DRDC@templena}
\newsavebox\DRDC@box
\newsavebox\DRDC@boxa
\newsavebox\DRDC@boxb
\newsavebox\DRDC@boxc
\newcommand\DRDC@up[1]{\raisebox{.16ex}{#1}}
\newcommand\DRDC@pup{\protect\DRDC@up}
\newcommand*\DRDC@SR{SR}
\newcommand*\DRDC@RD{RD}
\newcommand*\DRDC@U{U}
\let\DRDC@doccat\DRDC@SR
\DeclareOption{sr}{\let\DRDC@doccat\DRDC@SR}
\DeclareOption{rd}{\let\DRDC@doccat\DRDC@RD}
\DeclareOption{unpublished}{\let\DRDC@doccat\DRDC@U}
\newif\ifDRDC@isbrief
\DRDC@isbrieffalse
\newcommand\DRDC@checkfree[1]{%
  \ifx\DRDC@doccat\DRDC@U #1
  \else\DRDC@error{Option \CurrentOption\space can only be used for
    unpublished\MessageBreak reports}\fi}
\DeclareOption{a4paper}{%
  \DRDC@checkfree{\setlength\paperheight {297mm}%
                  \setlength\paperwidth  {210mm}}}
\DeclareOption{a5paper}{%
  \DRDC@checkfree{\setlength\paperheight {210mm}%
                  \setlength\paperwidth  {148mm}}}
\DeclareOption{b5paper}{%
  \DRDC@checkfree{\setlength\paperheight {250mm}%
                  \setlength\paperwidth  {176mm}}}
\DeclareOption{letterpaper}{%
  \setlength\paperheight {11in}%
  \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}{%
  \DRDC@checkfree{\setlength\paperheight {14in}%
                  \setlength\paperwidth  {8.5in}}}
\DeclareOption{executivepaper}{%
  \DRDC@checkfree{\setlength\paperheight {10.5in}%
                  \setlength\paperwidth  {7.25in}}}
\newcommand\@ptsize{1}
\DeclareOption{10pt}{\DRDC@checkfree{\renewcommand\@ptsize{0}}}
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\DRDC@checkfree{\renewcommand\@ptsize{2}}}
\newif\ifDRDC@isnumbered \DRDC@isnumberedtrue
\DeclareOption{plain}{\DRDC@isnumberedfalse}
\newif\ifDRDC@isfrench \DRDC@isfrenchfalse
\DeclareOption{french}{\DRDC@isfrenchtrue}
\DeclareOption{english}{\DRDC@isfrenchfalse}
\newif\ifDRDC@usecmmath\DRDC@usecmmathtrue
\newif\ifDRDC@usecmtext\DRDC@usecmtexttrue
\DeclareOption{psfonts}{\DRDC@usecmmathfalse\DRDC@usecmtextfalse}
\DeclareOption{pstextonly}{\DRDC@usecmtextfalse}
\newif\ifDRDC@compressedauthors
\DRDC@compressedauthorsfalse
\DeclareOption{compressedauthors}{\DRDC@compressedauthorstrue}
\DeclareOption{oneside}{%
  \DRDC@checkfree{\@twosidefalse\@mparswitchfalse}}
\DeclareOption{twoside}{\@twosidetrue\@mparswitchtrue}
\newif\ifDRDC@isdraft \DRDC@isdraftfalse
\DeclareOption{draft}{\setlength\overfullrule{5\p@}\DRDC@isdrafttrue}
\DeclareOption{final}{\setlength\overfullrule{\z@}\DRDC@isdraftfalse}
\newcommand\DRDC@nowatermark{none}
\newcommand\DRDC@normalwatermark{normal}
\newcommand\DRDC@marginwatermark{margin}
\let\DRDC@drafttype\DRDC@normalwatermark
\DeclareOption{nowatermark}{\let\DRDC@drafttype\DRDC@nowatermark}
\DeclareOption{marginwatermark}{%
  \let\DRDC@drafttype\DRDC@marginwatermark}
\DeclareOption{leqno}{\DRDC@checkfree{\input{leqno.clo}}}
\DeclareOption{fleqn}{\DRDC@checkfree{\input{fleqn.clo}}}
\newcommand{\DRDC@secpagebreak}{}
\DeclareOption{secpagebreak}{%
  \renewcommand{\DRDC@secpagebreak}{\clearpage}}
\newif\ifDRDC@bothtitles\DRDC@bothtitlesfalse
\DeclareOption{bothtitles}{\DRDC@bothtitlestrue}
\newif\ifDRDC@authorrefs\DRDC@authorrefsfalse
\DeclareOption{authorrefs}{\DRDC@authorrefstrue}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\newif\ifDRDC@pdfa\DRDC@pdfafalse
\DeclareOption{pdfa}{\DRDC@pdfatrue}
\@ifundefined{pdfpagewidth}{}{%
   \setlength\pdfpagewidth{\strip@pt\paperwidth truept}%
   \setlength\pdfpageheight{\strip@pt\paperheight truept}}
\ExecuteOptions{letterpaper,11pt,twoside}
\ProcessOptions\relax
\newif\ifDRDC@translationwarning\DRDC@translationwarningfalse
\ifDRDC@isfrench
  \newcommand\DRDC@translationerror{%
    \DRDC@error{There is standard text in this document that
      \MessageBreak has no official French translation.  It has
      \MessageBreak been highlighted using red text}{}}
  \newcommand\DRDC@needstranslation[1]{%
    \textcolor{red}{#1}%
  }
  \AtEndDocument{%
    \ifDRDC@translationwarning\DRDC@translationerror\fi}
\fi
\ifDRDC@pdfa\RequirePackage{drdc-pdfa}\fi
\RequirePackage[final]{graphics}
\RequirePackage{ifpdf}
\newcommand\DRDC@ext{\ifpdf pdf\else eps\fi}
\RequirePackage{exscale}
\ifx\DRDC@doccat\DRDC@U\else\RequirePackage{drdc-docctl}\fi
\ifDRDC@isfrench
  \newcommand\DRDC@draftname{BROUILLON}
\else
  \newcommand\DRDC@draftname{DRAFT}
\fi
\ifDRDC@isdraft
  \ifx\DRDC@drafttype\DRDC@nowatermark\else
    \RequirePackage{eso-pic}
    \newcommand\DRDC@watermark[4]{%
      \setlength{\unitlength}{1pt}%
      \put(\strip@pt#1,\strip@pt#2){%
        \makebox(0,0)[c]{\rotatebox{#3}{\textcolor[gray]{0.94}{%
          \fontsize{#4}{#4}{\sffamily\bfseries\DRDC@draftname}}}}}}
    \ifx\DRDC@drafttype\DRDC@marginwatermark
      \AddToShipoutPicture{%
        \begingroup
        \setlength{\DRDC@templen}{0.5in}%
        \addtolength{\DRDC@templen}{0.5\oddsidemargin}%
        \setlength{\DRDC@templena}{0.5\paperheight}%
        \DRDC@watermark{\DRDC@templen}{\DRDC@templena}{90}{1in}
        \setlength{\DRDC@templen}{-\DRDC@templen}%
        \addtolength{\DRDC@templen}{\paperwidth}%
        \DRDC@watermark{\DRDC@templen}{\DRDC@templena}{90}{1in}
        \endgroup
      }
    \else
      \newcommand\DRDC@draftsize{\ifDRDC@isfrench 1.5in\else 2in\fi}%
      \AddToShipoutPicture{%
        \begingroup
        \setlength{\DRDC@templen}{4.251in}%
        \setlength{\DRDC@templena}{0.5\paperheight}%
        \DRDC@watermark{\DRDC@templen}{\DRDC@templena}{45}
                       {\DRDC@draftsize}
        \endgroup
      }
    \fi
  \fi
\fi
\newcommand*\DRDC@DMCA{A}
\newcommand*\DRDC@DMCQ{Q}
\newcommand*\DRDC@DMCD{D}
\let\DRDC@maxDMC\DRDC@DMCA
\newcommand*\ControlledGoods[1]{
  \xdef\DRDC@maxDMC{#1}
  \ifx\DRDC@maxDMC\DRDC@DMCA\else
    \ifx\DRDC@maxDMC\DRDC@DMCQ\else
      \ifx\DRDC@maxDMC\DRDC@DMCD\else
        \DRDC@error{Argument of \noexpand\ControlledGoods
                    must be A, Q or D}%
      \fi
    \fi
  \fi
}
\AtBeginDocument{%
  \ifx\DRDC@maxDMC\DRDC@DMCA\else
    \@ifpackageloaded{drdc-classified}{}{
        \DRDC@error{Controlled Goods DMC is \DRDC@maxDMC\ but
                    \MessageBreak drdc-classified has not been loaded.
                    \MessageBreak Documents with this DMC must be at
                    least\MessageBreak PROTECTED A}}
  \fi}%
\ifDRDC@isfrench
  \newcommand*{\DRDC@DMCAterm}{MARCHANDISES NON CONTR\^{O}L\'{E}ES}
  \newcommand*{\DRDC@DMCQterm}{CONTR\^{O}LES \`{A} L'EXPORTATION}
  \newcommand*{\DRDC@DMCDterm}{MARCHANDISES CONTR\^{O}L\'{E}ES}
\else
  \newcommand*{\DRDC@DMCAterm}{NON-CONTROLLED GOODS}
  \newcommand*{\DRDC@DMCQterm}{EXPORT CONTROLLED}
  \newcommand*{\DRDC@DMCDterm}{CONTROLLED GOODS}
\fi
\newcommand*\DRDC@makeCGterm[2]{%
  \global\let#1\@empty
  \begin{DRDC@switch}{#2}%
  \case{\DRDC@DMCA}{\g@addto@macro#1{\DRDC@DMCAterm}}%
  \case{\DRDC@DMCQ}{\g@addto@macro#1{\DRDC@DMCQterm}}%
  \case{\DRDC@DMCD}{\g@addto@macro#1{\DRDC@DMCDterm}}%
  \end{DRDC@switch}}
\newcommand*{\DRDC@CGterm}[1]{%
  \DRDC@makeCGterm\DRDC@tempb{#1}\DRDC@tempb}
\newcommand*\GCECdate[1]{%
  \DRDC@error{The macro \noexpand\GCECdate is no longer used}}
\DeclareTextSymbolDefault{\textsterling}{T1}
\expandafter\let
  \csname OT1\string\textsterling\endcsname\@undefined
\renewcommand{\sfdefault}{phv}
\providecommand\phvscale{.95}
\def\Hv@scale{\phvscale}
\DeclareMathAlphabet{\mathsf}{OT1}{phv}{m}{n}
\SetMathAlphabet{\mathsf}{bold}{OT1}{phv}{b}{n}
\ifDRDC@usecmmath\else
  \DeclareSymbolFont{operators}   {OT1}{ptmcm}{m}{n}
  \DeclareSymbolFont{letters}     {OML}{ptmcm}{m}{it}
  \DeclareSymbolFont{symbols}     {OMS}{pzccm}{m}{n}
  \DeclareSymbolFont{largesymbols}{OMX}{psycm}{m}{n}
  \DeclareSymbolFont{bold}        {OT1}{ptm}{bx}{n}
  \DeclareSymbolFont{italic}      {OT1}{ptm}{m}{it}
  \DeclareMathSymbol{\omicron}{0}{operators}{`\o}
  \thinmuskip=2mu
  \medmuskip=2.5mu plus 1mu minus 1mu
  \thickmuskip=4mu plus 1.5mu minus 1mu
  \def\boldmath{%
     \@warning{there is no bold Symbol font}%
     \global\let\boldmath=\relax
  }
  \DeclareMathSizes{5}{5}{5}{5}
  \DeclareMathSizes{6}{6}{5}{5}
  \DeclareMathSizes{7}{7}{5}{5}
  \DeclareMathSizes{8}{8}{6}{5}
  \DeclareMathSizes{9}{9}{7}{5}
  \DeclareMathSizes{10}{10}{7.4}{6}
  \DeclareMathSizes{10.95}{10.95}{8}{6}
  \DeclareMathSizes{12}{12}{9}{7}
  \DeclareMathSizes{14.4}{14.4}{10.95}{8}
  \DeclareMathSizes{17.28}{17.28}{12}{10}
  \DeclareMathSizes{20.74}{20.74}{14.4}{12}
  \DeclareMathSizes{24.88}{24.88}{17.28}{14.4}
  \DeclareFontSubstitution{OMS}{pzccm}{m}{n}
  \begingroup
  \nfss@catcodes
  \DeclareFontFamily{OMS}{ptm}{}
  \DeclareFontShape{OMS}{ptm}{m}{n}{<-> ssub * pzccm/m/n}{}
  \DeclareFontShape{OMS}{ptm}{m}{it}{<-> ssub * pzccm/m/n}{}
  \DeclareFontShape{OMS}{ptm}{m}{sl}{<-> ssub * pzccm/m/n}{}
  \DeclareFontShape{OMS}{ptm}{m}{sc}{<-> ssub * pzccm/m/n}{}
  \DeclareFontShape{OMS}{ptm}{bx}{n}{<-> ssub * pzccm/b/n}{}
  \DeclareFontShape{OMS}{ptm}{bx}{it}{<-> ssub * pzccm/b/n}{}
  \DeclareFontShape{OMS}{ptm}{bx}{sl}{<-> ssub * pzccm/b/n}{}
  \DeclareFontShape{OMS}{ptm}{bx}{sc}{<-> ssub * pzccm/b/n}{}
  \endgroup
\fi
\normalfont
\input{size1\@ptsize.clo}
\ifDRDC@usecmtext\else
  \renewcommand{\rmdefault}{ptm}
  \renewcommand{\ttdefault}{pcr}
\fi
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{11\p@ \@plus2\p@ \@minus 1\p@}
\setlength\parindent{\z@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setlength\headheight{17\p@}
\setlength\headsep{0.5in}
\setlength\topmargin{-\headheight}
\addtolength\topmargin{-\headsep}
\setlength\footskip{0.5in}
\setlength\maxdepth{.5\topskip}
\newcommand\settextwidth[1]{%
  \setlength\textwidth{#1}%
  \setlength\DRDC@templen{\paperwidth}%
  \addtolength\DRDC@templen{-\textwidth}%
  \setlength\oddsidemargin{.5\DRDC@templen}%
  \addtolength\oddsidemargin{-1in}%
  \setlength\evensidemargin{\oddsidemargin}%
  \setlength\marginparwidth   {.5\DRDC@templen}%
  \addtolength\marginparwidth {-\marginparsep}%
  \addtolength\marginparwidth {-0.4in}%
  \@settopoint\textwidth
  \@settopoint\oddsidemargin
  \@settopoint\marginparwidth
  \@settopoint\evensidemargin
}
\setlength\DRDC@templen{\paperwidth}
\addtolength\DRDC@templen{-2.0true in}
\setlength\DRDC@templena{6.0true in}
\ifdim\DRDC@templen>\DRDC@templena\relax
  \settextwidth{\DRDC@templena}
\else
  \settextwidth{\DRDC@templen}
\fi
\setlength{\textheight}{21true cm}
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\ifDRDC@isfrench
  \newcommand\DRDC@blanktext{Page laiss\'ee en blanc \`a dessein.}
\else
  \newcommand\DRDC@blanktext{This page intentionally left blank.}
\fi
\newcommand\blankpage{
  \clearpage
  \vspace*{150pt}
  \begin{center}\normalfont\normalsize\DRDC@blanktext\end{center}
  \clearpage
}
\renewcommand\cleardoublepage{\clearpage\if@twoside
   \ifodd\c@page\else\blankpage\fi\fi}
\let\DRDC@headfootcontents\@empty
\newcommand\DRDC@thisheadfootcontents{\DRDC@headfootcontents}
\newif\ifDRDC@headfootchanged\DRDC@headfootchangedfalse
\newcommand\oneheadfoot[1]{%
  \gdef\DRDC@thisheadfootcontents{#1}%
  \global\DRDC@headfootchangedtrue
}
\newcommand\DRDC@resetheadfoot{%
  \ifDRDC@headfootchanged
    \gdef\DRDC@thisheadfootcontents{\DRDC@headfootcontents}%
    \global\DRDC@headfootchangedfalse
  \fi
}
\newcommand\defineheadfoot[1]{%
  \gdef\DRDC@headfootcontents{#1}%
  \DRDC@headfootchangedtrue\DRDC@resetheadfoot
}
\newcommand\thisheadfoot[2]{%
  \DRDC@headfootwarning\oneheadfoot{#1 #2}%
}
\newcommand\allheadfoot[2]{%
  \DRDC@headfootwarning\defineheadfoot{#1 #2}%
}
\newcommand\DRDC@headfootwarning{%
  \DRDC@error{\noexpand\thisheadfoot\ and \noexpand\allheadfoot\
  have been replaced \MessageBreak by \noexpand\oneheadfoot\ and
  \noexpand\defineheadfoot\ each with only one argument}%
}

\newcommand*\headfootsize{\small}
\newcommand*\DRDC@headfootformat{\headfootsize\sffamily\bfseries}
\newcommand\DRDC@headfootprehook{}
\newlength\headfootwidth
\setlength\headfootwidth{\textwidth}
\newcommand\DRDC@classout{%
  \DRDC@headfootprehook
  \parbox[t]{\textwidth}{\centering
    \parbox[t]{\headfootwidth}{\DRDC@headfootformat\centering
    \DRDC@thisheadfootcontents}}}
\let\DRDC@adddraftheadfoot\@empty
\let\DRDC@adddraftthisheadfoot\@empty
\ifDRDC@isdraft\ifx\DRDC@drafttype\DRDC@nowatermark
  \renewcommand\DRDC@adddraftheadfoot{%
    \ifx\DRDC@headfootcontents\@empty
      \def\DRDC@headfootcontents{\DRDC@draftname}%
    \else
      \g@addto@macro\DRDC@headfootcontents{\\\DRDC@draftname}%
    \fi}
  \renewcommand\DRDC@adddraftthisheadfoot{%
    \ifx\DRDC@thisheadfootcontents\@empty
      \def\DRDC@thisheadfootcontents{\DRDC@draftname}%
    \else
      \g@addto@macro\DRDC@thisheadfootcontents{\\\DRDC@draftname}%
    \fi}
  \DRDC@adddraftheadfoot
\fi\fi
\newcommand*\DRDC@pagenumformat{\small\sffamily}
\newcommand*\DRDC@repnumformat{\DRDC@pagenumformat}
\newlength{\DRDC@footlineheight}
\sbox\DRDC@box{\DRDC@headfootformat\`{A}}
\setlength\DRDC@footlineheight{\ht\DRDC@box}
\newlength{\DRDC@footheight}
\setlength\DRDC@footheight{\paperheight}
\addtolength\DRDC@footheight{-1.0in}
\addtolength\DRDC@footheight{-\textheight}
\addtolength\DRDC@footheight{-0.9cm}
\addtolength\DRDC@footheight{-\footskip}
\addtolength\DRDC@footheight{\DRDC@footlineheight}
\newlength\DRDC@footlinesize
\newcommand\DRDC@footbox[2]{%
  \parbox[t][\DRDC@footheight][t]{\textwidth}{%
    \rule{0pt}{\DRDC@footlineheight}#1\vfill #2}%
  \DRDC@resetheadfoot}
\def\ps@empty{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{\DRDC@classout}%
  \def\@evenhead{\@oddhead}%
  \def\@oddfoot{\DRDC@footbox{}{\DRDC@classout}}%
  \def\@evenfoot{\@oddfoot}}%
\def\ps@cover{%
  \let\@mkboth\@gobbletwo
  \def\@oddhead{%
    \parbox[t]{\textwidth}{%
      \hspace*{-0.5in}\scalebox{0.24979}{%
        \includegraphics{%
          DRDC-FIP\ifDRDC@isfrench -french\fi.\DRDC@ext}}%\\[5\p@]
      \\[31\p@]
      \sbox\DRDC@box{\includegraphics{boxes.\DRDC@ext}}%
      \sbox\DRDC@boxa{%
         \resizebox{3.4cm}{!}{\raisebox{-0.3271\ht\DRDC@box}{%
           \usebox\DRDC@box}}}%
      \dp\DRDC@boxa=\z@\ht\DRDC@boxa=7\p@\wd\DRDC@boxa=1.384cm%
      \setlength\DRDC@templen{\textwidth}%
      \addtolength\DRDC@templen{-\wd\DRDC@boxa}%
      \resizebox{\DRDC@templen}{8.5\p@}{%
        \includegraphics{rule.\DRDC@ext}}\usebox\DRDC@boxa\\[10pt]%
      \hspace*{9cm}\resizebox{4.8cm}{!}{%
        \includegraphics{DRDC-RDDC-tech.\DRDC@ext}}\par
     \vspace*{-100pt}%
     {\setlength\headfootwidth{285pt}%
      \parbox[t][42pt][c]{\textwidth}{\DRDC@classout}}%
    }%
  }%
  \let\@evenhead\@empty
  \def\@oddfoot{%
    \sbox\DRDC@boxb{\hspace*{1.45cm}\resizebox{3.0cm}{!}{%
        \includegraphics{wordmark.\DRDC@ext}}}%
    \wd\DRDC@boxb=1.51cm\ht\DRDC@boxb=\z@
    {\setlength\headfootwidth{285pt}%
      \headfootsize
    %  \sbox\DRDC@boxc{\parbox[b]{\textwidth}{\DRDC@classout}}%
    \sbox\DRDC@boxc{\mbox{\raisebox{40pt}{\DRDC@classout}}}%
      \sbox\DRDC@boxa{\llap{\usebox{\DRDC@boxb}}%
        \raisebox{\dp\DRDC@boxc}{\usebox{\DRDC@boxc}}}%
      \DRDC@footbox{\mbox{}}{\usebox{\DRDC@boxa}}%
    }%
    \DRDC@resetheadfoot}%
  \let\@evenfoot\@empty
}
\if@twoside
  \def\ps@headings{%
    \def\@oddhead{\DRDC@classout}%
    \let\@evenhead\@oddhead
    \def\@evenfoot{\DRDC@footbox{%
       {\DRDC@pagenumformat\thepage}\hfill
       {\DRDC@repnumformat\DRDC@repnumber}}{\DRDC@classout}}%
    \def\@oddfoot{\DRDC@footbox{%
       {\DRDC@repnumformat\DRDC@repnumber}\hfill
       {\DRDC@pagenumformat\thepage}}{\DRDC@classout}}%
    \let\@mkboth\@gobbletwo}%
\else
  \def\ps@headings{%
    \def\@oddhead{\DRDC@classout}%
    \def\@oddfoot{\DRDC@footbox{%
       {\DRDC@repnumformat\DRDC@repnumber}\hfill
       {\DRDC@pagenumformat\thepage}}{\DRDC@classout}}%
    \let\@mkboth\@gobbletwo}%
\fi
\renewcommand\ps@plain{\ps@headings}
\newcommand\ps@myheadings{\ps@headings}
\ifx\DRDC@doccat\DRDC@U
  \newcommand\ps@blank{%
    \let\@oddhead\@empty
    \let\@evenhead\@empty
    \let\@oddfoot\@empty
    \let\@evenfoot\@empty
    \let\@mkboth\@gobbletwo}
  \newcommand\ps@norepnumber{%
    \def\@oddhead{\DRDC@classout}%
    \let\@evenhead\@oddhead
    \def\@oddfoot{\DRDC@footbox{%
      \makebox[\textwidth]{\DRDC@pagenumformat\thepage}}%
      {\DRDC@classout}}%
    \let\@evenfoot\@oddfoot
    \let\@mkboth\@gobbletwo}
\fi
\let\DRDC@subtitle\@empty
\newcommand\subtitle[1]{\renewcommand*{\DRDC@subtitle}{#1}}
\let\DRDC@authors\@empty
\newcommand\authors[1]{\renewcommand*{\DRDC@authors}{#1}}
\let\@author\@empty
\newcommand\DRDC@defaultauthors{%
  \ifx\DRDC@authors\@empty
     \ifx\@author\@empty
        \DRDC@error{Authors undefined}%
     \else
        \let\DRDC@authors\@author
     \fi
  \fi
}
\long\def\DRDC@parseauthors#1#2{%
  \def\DRDC@typesetauthor{#1}%
  \long\def\DRDC@authordelim{#2}%
  \DRDC@defaultauthors
  \expandaftertwo\DRDC@firstspaceanditem\DRDC@authors\space\and\@nil\@@%
}
\long\def\DRDC@firstspaceanditem#1 \and#2\@@{%
  \DRDC@firstanditem#1\and\@nil\@@%
  \def\DRDC@temp{#2}%
  \ifx\DRDC@temp\@nnil\relax%
  \else\DRDC@authordelim\expandafter\DRDC@firstspaceanditem #2\@@\fi%
}
\newcommand\expandaftertwo[1]{%
  \expandafter\expandafter\expandafter#1\expandafter}
\long\def\DRDC@firstanditem#1\and#2\@@{%
  \DRDC@oneanditem#1[]\@nil\@@%
  \def\DRDC@temp{#2}%
  \ifx\DRDC@temp\@nnil\relax%
  \else\DRDC@authordelim\expandafter\DRDC@firstanditem #2\@@\fi%
}
\def\DRDC@bnil{[]\@nil}
\def\DRDC@spbnil{ []\@nil}
\long\def\DRDC@oneanditem#1[#2]#3\@@{%
  \DRDC@typesetauthor{#1}{#2}%
  \def\DRDC@temp{#3}%
  \ifx\DRDC@temp\@nnil\relax
  \else\ifx\DRDC@temp\DRDC@bnil\relax
  \else\ifx\DRDC@temp\DRDC@spbnil\relax
  \else\DRDC@authordelim\expandafter\DRDC@oneanditem #3\@@\fi\fi\fi
}
\newcommand*\rank[1]{#1}
\newcommand\DRDC@addtoauthors[2]{%
  \g@addto@macro\DRDC@authorlist{\DRDC@cmd{#1}{#2}}}%
\newcommand\DRDC@makeauthorlist{%
  \def\DRDC@authorlist{}%
  \DRDC@parseauthors\DRDC@addtoauthors{}}%
\newif\ifDRDC@firstitem
\newcommand\makeexpandedauthors[2]{%
  \begingroup
    \DRDC@makeauthorlist
    \def~{ }%
    \DRDC@firstitemtrue
    \def\DRDC@cmd##1##2{%
      \ifDRDC@firstitem\g@addto@macro\DRDC@temp{##1}%
      \else\g@addto@macro\DRDC@temp{#2##1}\fi
      \DRDC@firstitemfalse}%
    \def\DRDC@temp{}%
    \DRDC@authorlist
    \xdef#1{\DRDC@temp}%
  \endgroup}
\newif\ifDRDC@addcomma
\newcommand\DRDC@onetitleauthor[2]{%
  \def\DRDC@temp{#2}%
  \parskip\z@
  \ifDRDC@compressedauthors
    \ifDRDC@addcomma, \fi \mbox{\ignorespaces #1}%
    \ifx\DRDC@temp\@empty\DRDC@addcommatrue
    \else\par #2\par\vspace{1ex}\DRDC@addcommafalse\fi
  \else
    #1\par%
    \ifx\DRDC@temp\@empty\else #2\par\vspace{1ex}\fi
  \fi}
\newcommand\DRDC@titleauthors{%
  \DRDC@addcommafalse
  \DRDC@parseauthors{\DRDC@onetitleauthor}{}}
\gdef\DRDC@year{\year}
\gdef\DRDC@month{\month}
\gdef\DRDC@day{\day}
\newcommand\setdate[3]{%
  \gdef\DRDC@year{#1}%
  \gdef\DRDC@month{#2}%
  \gdef\DRDC@day{#3}%
}
\renewcommand\date[1]{%
  \DRDC@error{The \noexpand\date macro does not work with this
              class.\MessageBreak
              Use \noexpand\setdate instead}{}}
\newcommand\DRDC@montherror{%
  \DRDC@error{Unknown month.  Should be 1-12 in \noexpand\setdate}}
\newcommand*\numtomonth[1]{%
  \ifDRDC@isfrench\numtofmonth{#1}\else\numtoemonth{#1}\fi}
\newcommand*\numtoemonth[1]{\ifcase#1\DRDC@montherror\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\else
  \DRDC@montherror\fi}
\newcommand*\numtofmonth[1]{\ifcase#1\DRDC@montherror\or
  janvier\or f\'evrier\or mars\or avril\or mai\or juin\or
  juillet\or ao\^ut\or septembre\or octobre\or novembre\or
  d\'ecembre\else\DRDC@montherror\fi}
\newcommand*\NumToFMonth[1]{\ifcase#1\DRDC@montherror\or
  Janvier\or F\'evrier\or Mars\or Avril\or Mai\or Juin\or
  Juillet\or Ao\^ut\or Septembre\or Octobre\or Novembre\or
  D\'ecembre\else\DRDC@montherror\fi}
\newcommand\DRDC@fday[1]{\ifnum#1=1\relax 1\ier\else #1\fi}
\newcommand\DRDC@FDate{%
  \ifDRDC@isdraft\DRDC@fday{\number\DRDC@day}~\numtofmonth{\DRDC@month}%
  \else\NumToFMonth{\DRDC@month}\fi
  \space\number\DRDC@year
}
\newcommand\DRDC@edate{%
  \numtoemonth{\DRDC@month}%
  \ifDRDC@isdraft~\number\DRDC@day ,\fi
  \space\number\DRDC@year
}
\newcommand\DRDC@fdate{%
  \ifDRDC@isdraft\DRDC@fday{\number\DRDC@day}~\fi
  \numtofmonth{\DRDC@month}\space\number\DRDC@year
}
\ifDRDC@isfrench
  \renewcommand\@date{\DRDC@fdate}
  \newcommand\DRDC@Date{\DRDC@FDate}
\else
  \renewcommand\@date{\DRDC@edate}
  \newcommand\DRDC@Date{\DRDC@edate}
\fi
\ifx\DRDC@doccat\DRDC@U
  \newcommand\DRDC@repnumber{\mbox{}}
\else
  \newcommand\DRDC@repnumber{%
    \DRDC@error{No \noexpand\repnumber or \noexpand\docnumber given}}
\fi
\newcommand\repnumber[1]{\renewcommand*{\DRDC@repnumber}{#1}}
\let\DRDC@translated\@empty
\newcommand\translated{%
  \renewcommand\DRDC@translated{%
    \DRDC@pup(\ifDRDC@isfrench F\else E\fi\DRDC@pup)}}
\newcommand\DRDC@defrepnumber[2]{%
  DRDC-RDDC-\number\DRDC@year-#1#2\DRDC@translated}
\newcommand\DRDC@docletter{%
  \ifx\DRDC@doccat\DRDC@SR R\else D\fi
}
\newcommand\DRDC@docnumber{%
  \DRDC@error{No \noexpand\docnumber given}%
}
\newcommand\docnumber[1]{%
  \renewcommand*{\DRDC@docnumber}{#1}%
  \renewcommand\DRDC@repnumber{%
    \DRDC@defrepnumber{\DRDC@docletter}{\DRDC@docnumber}}%
}
\newcommand*\DRDC{Defence Research and Development Canada}
\newcommand*\RDDC{Recherche et d\'eveloppement pour la d\'efense Canada}
\newcommand*\DRDCA{DRDC -- Atlantic Research Centre}
\newcommand*\RDDCA{RDDC -- Centre de recherches de l'Atlantique}
\newcommand*\DRDCO{DRDC -- Ottawa Research Centre}
\newcommand*\RDDCO{RDDC -- Centre de recherches d'Ottawa}
\newcommand*\DRDCS{DRDC -- Suffield Research Centre}
\newcommand*\RDDCS{RDDC -- Centre de recherches de Suffield}
\newcommand*\DRDCT{DRDC -- Toronto Research Centre}
\newcommand*\RDDCT{RDDC -- Centre de recherches de Toronto}
\newcommand*\DRDCV{DRDC -- Valcartier Research Centre}
\newcommand*\RDDCV{RDDC -- Centre de recherches de Valcartier}
\newcommand*\DRDCCORA{DRDC -- Centre for Operational Research and
                              Analysis}
\newcommand*\RDDCCARO{RDDC -- Centre d'analyse et de recherche
                              op\'erationelle }
\newcommand*\DRDCCSS{DRDC -- Centre for Security Science}
\newcommand*\RDDCCSS{RDDC -- Centre des sciences pour la s\'ecurit\'e}
\newcommand*\DRDCDGMPRA{DRDC -- Director General Military
                                Personnel Research and Analysis}
\newcommand*\RDDCDGRAPM{RDDC -- Directeur g\'en\'eral --
                                Recherche et analyse
                                (Personnel militaire)}
\ifDRDC@isfrench
  \newcommand*\DRDCAaddress{%
    CP~1012, Dartmouth NS~~B2Y~3Z7, Canada}
  \newcommand*\DRDCOaddress{%
    3701, avenue Carling, Ottawa ON~~K1A~0Z4, Canada}
  \newcommand*\DRDCSaddress{%
    CP~4000, Succ.\ Main, Medicine Hat AB~~T1A~8K6, Canada}
  \newcommand*\DRDCTaddress{%
    1133, avenue Sheppard ouest, Toronto ON~~M3K~2C9, Canada}
  \newcommand*\DRDCVaddress{%
    2459, route de la Bravoure, Qu\'ebec QC~~G3J~1X5, Canada}
  \newcommand*\DRDCCORAaddress{%
    101, prom Colonel By, Ottawa ON~~K1A~0K2, Canada}
  \newcommand*\DRDCCSSaddress{%
    222, rue Nepean, Ottawa, ON~~K1A~0K2, Canada}
  \newcommand*\DRDCDGMPRAaddress{%
    285, chemin Coventry, Ottawa, ON~~K1A~0K2, Canada}
  \newcommand*\DRDCaddress{%
    400, rue Cumberland, Ottawa ON~~K1N~8X3, Canada}
\else
  \newcommand*\DRDCAaddress{%
    PO~Box 1012, Dartmouth NS~~B2Y~3Z7, Canada}
  \newcommand*\DRDCOaddress{%
    3701 Carling Avenue, Ottawa ON~~K1A~0Z4, Canada}
  \newcommand*\DRDCSaddress{%
    Box~4000, Station Main, Medicine Hat AB~~T1A~8K6, Canada}
  \newcommand*\DRDCTaddress{%
    1133 Sheppard Avenue West, Toronto ON~~M3K~2C9, Canada}
  \newcommand*\DRDCVaddress{%
    2459 de la Bravoure Road, Qu\'ebec QC~~G3J~1X5, Canada}
  \newcommand*\DRDCCORAaddress{%
   Carling Campus, 60 Moodie Drive, building 7S.2,
Kanata, ON~~K1A~0K2, Canada}
  \newcommand*\DRDCCSSaddress{%
    Carling Campus, 60 Moodie Drive, building 7,
Kanata, ON~~K1A~0K2, Canada}
  \newcommand*\DRDCDGMPRAaddress{%
    Carling Campus, 60 Moodie Drive, building 9S.2,
Kanata, ON~~K1A~0K2, Canada}
  \newcommand*\DRDCaddress{%
    400 Cumberland Street, Ottawa ON~~K1N~8X3, Canada}
\fi
\newcommand\DRDC@eestablishmentname{%
  \DRDC@error{No \noexpand\establishment given}}
\newcommand\DRDC@festablishmentname{%
  \DRDC@error{No \noexpand\festablishment given}}
\newcommand*\DRDC@establishmentaddress{%
  \DRDC@error{No \noexpand\establishmentaddress given}}
\ifDRDC@isfrench
  \newcommand\establishmentname{\DRDC@festablishmentname}%
  \newcommand\DRDC@agencyname{\RDDC}%
\else
  \newcommand\establishmentname{\DRDC@eestablishmentname}%
  \newcommand\DRDC@agencyname{\DRDC}%
\fi
\newcommand\establishment[1]{%
  \renewcommand*{\DRDC@eestablishmentname}{#1}%
  \def\DRDC@tempb##1##2##3{%
    \gdef\DRDC@eestablishmentname{##1}%
    \gdef\DRDC@festablishmentname{##2}%
    \establishmentaddress{##3}}%
  \begin{DRDC@switch}{#1}
  \case{\DRDCA}{\DRDC@tempb{\DRDCA}{\RDDCA}{\DRDCAaddress}}
  \case{\DRDCO}{\DRDC@tempb{\DRDCO}{\RDDCO}{\DRDCOaddress}}
  \case{\DRDCS}{\DRDC@tempb{\DRDCS}{\RDDCS}{\DRDCSaddress}}
  \case{\DRDCT}{\DRDC@tempb{\DRDCT}{\RDDCT}{\DRDCTaddress}}
  \case{\DRDCV}{\DRDC@tempb{\DRDCV}{\RDDCV}{\DRDCVaddress}}
  \case{\DRDCCORA}{%
    \DRDC@tempb{\DRDCCORA}{\RDDCCARO}{\DRDCCORAaddress}}
  \case{\DRDCCSS}{\DRDC@tempb{\DRDCCSS}{\RDDCCSS}{\DRDCCSSaddress}}
  \case{\DRDCDGMPRA}{%
    \DRDC@tempb{\DRDCDGMPRA}{\RDDCDGRAPM}{\DRDCDGMPRAaddress}}%
  \case{\DRDC}{\DRDC@tempb{\DRDC}{\RDDC}{\DRDCaddress}}
  \case{\RDDCA}{\DRDC@tempb{\DRDCA}{\RDDCA}{\DRDCAaddress}}
  \case{\RDDCO}{\DRDC@tempb{\DRDCO}{\RDDCO}{\DRDCOaddress}}
  \case{\RDDCS}{\DRDC@tempb{\DRDCS}{\RDDCS}{\DRDCSaddress}}
  \case{\RDDCT}{\DRDC@tempb{\DRDCT}{\RDDCT}{\DRDCTaddress}}
  \case{\RDDCV}{\DRDC@tempb{\DRDCV}{\RDDCV}{\DRDCVaddress}}
  \case{\RDDCCARO}{%
    \DRDC@tempb{\DRDCCORA}{\RDDCCARO}{\DRDCCORAaddress}}
  \case{\RDDCCSS}{\DRDC@tempb{\DRDCCSS}{\RDDCCSS}{\DRDCCSSaddress}}
  \case{\RDDCDGRAPM}{%
    \DRDC@tempb{\DRDCDGMPRA}{\RDDCDGRAPM}{\DRDCDGMPRAaddress}}%
  \case{\RDDC}{\DRDC@tempb{\DRDC}{\RDDC}{\DRDCaddress}}
  \default{\DRDC@error{Non-standard Research Centre names no longer
                       \MessageBreak allowed}}
  \end{DRDC@switch}%
}
\newcommand\festablishment{\establishment}
\newcommand\establishmentaddress[1]{%
  \gdef\DRDC@establishmentaddress{#1}}
\let\DRDC@client\@empty
\let\DRDC@clientaddress\@empty
\newcommand\preparedfor[2]{%
  \renewcommand*{\DRDC@client}{#1}%
  \renewcommand*{\DRDC@clientaddress}{#2}%
  \DRDC@addECRstatements
}
\newcommand\preparedforsep{\\}
\newcommand\DRDC@preparedfor{%
  \DRDC@client\preparedforsep\DRDC@clientaddress}
\ifDRDC@isfrench
  \newcommand\DRDC@preparedfortext{R\'edig\'e pour le compte de}
\else
  \newcommand\DRDC@preparedfortext{Prepared for}
\fi
\newcommand\DRDC@defaddtomacros[1]{
  \expandafter\def\csname DRDC@#1\endcsname{}%
  \expandafter\def\csname #1\endcsname ##1{%
    \AtBeginDocument{%
      \expandafter\renewcommand\csname DRDC@#1\endcsname{}%
      \csname DRDC@addto#1\endcsname{##1}}}%
  \expandafter\def\csname addto#1\endcsname ##1{%
    \AtBeginDocument{\csname DRDC@addto#1\endcsname{##1}}}%
  \expandafter\def\csname DRDC@addto#1\endcsname ##1{%
    \def\DRDC@temp{}%
    \expandafter\ifx\csname DRDC@#1\endcsname\DRDC@temp
      \expandafter\def\csname DRDC@#1\endcsname{##1}%
    \else
      \expandafter\g@addto@macro\csname DRDC@#1\endcsname{\par ##1}%
    \fi}%
}
\DRDC@defaddtomacros{sensitivitystatements}
\DRDC@defaddtomacros{IPstatements}
\DRDC@defaddtomacros{releasestatements}
\DRDC@defaddtomacros{warranties}
\DRDC@defaddtomacros{disclaimers}
\DRDC@defaddtomacros{infostatements}
\DRDC@defaddtomacros{admininfo}
\DRDC@defaddtomacros{standardclauses}
\newcommand\protectivestatements{\infostatements}
\addtoinfostatements{The body of this CAN UNCLASSIFIED document does not contain the required security banners according to DND security
standards. However, it must be treated as CAN UNCLASSIFIED and protected appropriately based on the terms and conditions
specified on the covering page.}

\newcommand\humansubjectsclause{%
  \addtostandardclauses{The data collected as part of this study was
    approved either by \DRDC's Human Research Ethics Board or by the
    Director General Military Personnel Research \& Analysis' Social
    Science Research Review Board.}}
\newcommand\animalsubjectsclause{%
  \addtostandardclauses{In conducting the research described in this
    report, the investigators adhered to the `Guide to the Care and
    Use of Experimental Animals, Vol.~I, 2\textsuperscript{nd}~Ed.'
    published by the Canadian Council on Animal Care.}}
\ifDRDC@isfrench
\newcommand*\DRDC@prtext{r\'evis\'ee par les pairs et }
\else
\newcommand*\DRDC@prtext{peer-reviewed and }
\fi
\newcommand\DRDC@pr{\ifx\DRDC@doccat\DRDC@SR\DRDC@prtext\fi}
\newcommand\peerreviewed{%
 \renewcommand\DRDC@pr{\DRDC@prtext}
}
\ifx\DRDC@doccat\DRDC@U\else
  \ifDRDC@isfrench
      \addtodisclaimers{%
     
 Avis de non-responsabilit\'e\DRDC@colon\ La pr\'esente publication a \'et\'e pr\'epar\'ee par Recherche et d\'eveloppement pour la d\'efense Canada, une agence du minist\`ere de la D\'efense nationale. Les renseignements qui figurent dans la pr\'esente publication d\'erivent de pratiques exemplaires et respectent  les plus hauts niveaux de la conduite responsable de la recherche scientifique. Ces renseignements sont destin\'es \`a l'usage du minist\`ere de la D\'efense nationale en partenariat avec les Forces arm\'ees canadiennes et S\'ecurité publique Canada et, tel que permis, peuvent \^etre divulgu\'es au milieu universitaire, \`a l'industrie, aux alli\'es du Canada et au grand public (``tiers").  Toute utilisation faite de la pr\'esente publication ou r\'ef\'erence \`a cette derni\`ere, ou d\'ecision prise sur la foi de celle-ci par une tierce partie, l'est  \`a ses propres risques et sous son enti\`ere responsabilit\'e. Le Canada n'assume aucune responsabilit\'e concernant les \'eventuels dommages ou pertes qui pourraient r\'esulter de toute utilisation de la publication ou de r\'ef\'erence \`a celle-ci.
%Sa Majest\'e la Reine du chef du Canada, repr\'esent\'ee par le ministre de la D\'efense nationale (<< le Canada >>) ne fait aucune d\'eclaration ni n'\'emet aucune garantie, explicite ou implicite, de quelque nature
%que ce soit et n'est pas responsable de l'exactitude, de la fiabilit\'e, de l'int\'egralit\'e, de la pertinence ou de l'utilit\'e de l'information, des produits, des processus et du mat\'eriel inclus dans ce document. Aucune information contenue dans ce document ne doit \^etre consid\'er\'ee comme une approbation pour l'usage sp\'ecifique de tout outil, technique ou processus examin\'e dans celui-ci. Le risque \'eventuel associ\'e \`a l'utilisation de l'information, des produits, des processus ou du mat\'eriel inclus dans ce document sera assum\'e enti\`erement par la personne qui en fait l'utilisation. Le Canada n'assume aucune responsabilit\'e quant aux dommages ou aux pertes qui d\'ecoulent de l'utilisation de l'information, des produits, des processus et du mat\'eriel inclus dans ce document.}
  \else
    \addtodisclaimers{%
      Disclaimer\DRDC@colon\ This publication was prepared by Defence Research and Development Canada an agency of the Department of National Defence. The information contained in this publication has been derived and determined through best practice and adherence to the highest standards of responsible conduct of scientific research. This information is intended for the use of the Department of National Defence, the Canadian Armed Forces (``Canada") and Public Safety partners and, as permitted, may be shared with academia, industry, Canada's allies, and the public (``Third Parties").  Any use by, or any reliance on or decisions made based on this publication by Third Parties, are done at their own risk and responsibility. Canada does not assume any liability for any damages or losses which may arise from any use of, or reliance on, the publication.}
  \fi
\fi
\ifx\DRDC@doccat\DRDC@U\else
  \ifDRDC@isfrench
    \newcommand\SLe{\ifx\DRDC@doccat\DRDC@SL e\fi}
    \newcommand\DRDC@endorsement{%
     \'Enonc\'e d'approbation\DRDC@colon\ La pr\'esente publication a \'et\'e \DRDC@pr
      publi\'ee par le Bureau de la r\'edaction de Recherche et
      d\'eveloppement pour la d\'efense Canada, un organisme du
      minist\`ere de la D\'efense nationale du Canada. Les demandes peuvent \^etre envoy\'ees \`a l'adresse suivante\DRDC@colon
      Publications.DRDC-RDDC@drdc-rddc.gc.ca.}
  \else
    \newcommand\DRDC@endorsement{%
      Endorsement statement\DRDC@colon\ This publication has been \DRDC@pr
      published by the Editorial Office of  Defence Research and
      Development Canada, an agency of the Department of National
      Defence of Canada. Inquiries can be sent to:
      Publications.DRDC-RDDC@drdc-rddc.gc.ca.}
  \fi
  \addtodisclaimers{\DRDC@endorsement}
\fi

\ifDRDC@isfrench
  \newcommand\DRDC@termsofrelease{%
    \textbf{Conditions de diffusion\DRDC@colon} Ce document est
    approuv\'e pour \^etre envoy\'e \`a \DRDC@recipients.  Toute
    distribution ult\'erieure de ce document ou de l'information
    incluse est interdite sans l'approbation \'ecrite de
    \RDDC\ (RDDC).}
\else
  \newcommand\DRDC@termsofrelease{%
    \textbf{Terms of Release:} This document is approved for
    release to \DRDC@recipients.  Further distribution of
    this document or information contained herein is
    prohibited without the written approval of \DRDC\ (DRDC).}
\fi
\newcommand*\recipients[1]{\gdef\DRDC@recipients{#1}}
\newcommand*\DRDC@recipients{%
  \DRDC@error{No \noexpand\recipients given}}
\newcommand\DRDC@CGadvisorystatement{This document was reviewed for Controlled Goods by DRDC using the Schedule to the \textit{Defence Production Act.}}
\newcommand\DRDC@CGhandlingstatement{}
\let\DRDC@IPowner\@empty
\newif\ifDRDC@protectedIP \DRDC@protectedIPfalse
\newcommand\protectIP[1][]{%
  \DRDC@error{\noexpand\protectIP requires the document to be
  \MessageBreak classified or protected}}
\let\DRDC@IPtype\@empty
\newcommand\DRDC@crownIP{crown}
\newcommand\DRDC@jointIP{joint}
\newcommand\DRDC@clientIP{client}
\newcommand\DRDC@IPerror{%
  \DRDC@error{Owner of intellectual property not indicated: use
              \MessageBreak\noexpand\crownIP, \noexpand\clientIP or
              \noexpand\jointIP}}
\let\DRDC@ECRstatement\@empty
\AtBeginDocument{%
  \ifx\DRDC@client\@empty\else
    \ifx\DRDC@ECRstatement\@empty\DRDC@IPerror\fi
  \fi
}
\ifDRDC@isfrench
  \newcommand\crownIP{
    \let\DRDC@IPtype\DRDC@crownIP
    \renewcommand\DRDC@ECRstatement{
      Cette publication est prot\'eg\'ee par droit d'auteur. Toute
      publication, utilisation ou diffusion de ce document est
      strictement interdite sans l'autorisation expresse de
      \RDDC. \DRDC@client\ est par la pr\'esente autoris\'e \`a
      utiliser et \`a autoriser l'utilisation de l'information
      contenue dans ce rapport pour ses propres besoins.
    }
  }
  \newcommand\clientIP{
    \let\DRDC@IPtype\DRDC@clientIP
    \renewcommand\DRDC@ECRstatement{
      Cette publication est prot\'eg\'ee par droit d'auteur. Toute
      publication, utilisation ou diffusion de ce document est
      strictement interdite sans l'autorisation expresse de
      \DRDC@client. Le minist\`ere de la D\'efense nationale du Canada
      doit d\'etenir une licence libre de redevances l'autorisant \`a
      utiliser et \`a autoriser l'utilisation des renseignements
      prot\'eg\'es par droit d'auteur pour ses besoins internes, \`a
      condition de ne pas les utiliser \`a des fins commerciales.
    }
  }
  \newcommand\jointIP{
    \let\DRDC@IPtype\DRDC@jointIP
    \renewcommand\DRDC@ECRstatement{
      Cette publication est prot\'eg\'ee par droit d'auteur. Toute
      publication, utilisation ou diffusion de ce document est
      strictement interdite sans l'autorisation expresse de
      \RDDC\ et de \DRDC@client.
    }
  }
  \newcommand\DRDC@ECRcommon{
    Ce rapport ne constitue pas une d\'eclaration d'ent\'erinement
    de la part du minist\`ere de la D\'efense nationale ou du
    gouvernement du Canada.}
\else
  \newcommand\crownIP{
    \let\DRDC@IPtype\DRDC@crownIP
    \renewcommand\DRDC@ECRstatement{
      This publication is protected by copyright.
      Any publication, use or release of this document is strictly
      prohibited without the express written permission of Her
      Majesty the Queen in Right of Canada.  \DRDC@client\ is hereby
      granted the right to use and have used information contained
      in this report for its own purposes.}
  }
  \newcommand\clientIP{
    \let\DRDC@IPtype\DRDC@clientIP
    \renewcommand\DRDC@ECRstatement{
      This publication is protected by copyright.  Any publication,
      use or release of this document is strictly prohibited without
      the expressed permission of \DRDC@client.  The Department of
      National Defence of Canada shall have a royalty-free right to
      use and have used any copyright information for its own internal
      purposes excluding any commercial use of the information.}
  }
  \newcommand\jointIP{
    \let\DRDC@IPtype\DRDC@jointIP
    \renewcommand\DRDC@ECRstatement{
      This publication is protected by copyright.  Any publication,
      use or release of this document is strictly prohibited without
      the express written permission of Her Majesty the Queen in
      Right of Canada and \DRDC@client.}
  }
  \newcommand\DRDC@ECRcommon{
    This report is not a statement of endorsement by the Department of
    National Defence or the Government of Canada.}
\fi
\newcommand\DRDC@addECRstatements{%
  \addtoIPstatements{\DRDC@ECRstatement}%
  \addtodisclaimers{\DRDC@ECRcommon}%
}
\let\DRDC@CGnotice\@empty
\def\DRDCcopyright{\raisebox{0.16ex}{\copyright}}
\newcommand\DRDC@copyrighttext{%
  \@hangfrom{\DRDCcopyright~}Her Majesty the Queen in Right of Canada,
  Department of National Defence, \number\DRDC@year

  \@hangfrom{\DRDCcopyright~}Sa Majest\'e la Reine en droit du Canada,
  Minist\`ere de la D\'efense nationale, \number\DRDC@year
}
\begin{DRDC@switch}{\DRDC@doccat}
\case{\DRDC@SR}{\xdef\DRDC@doccategory{%
    \ifDRDC@isfrench Rapport scientifique\else Scientific Report\fi}}
\case{\DRDC@RD}{\gdef\DRDC@doccategory{%
    \ifDRDC@isfrench Document de r\'ef\'erence\else
                     Reference Document\fi}}
\default{\gdef\DRDC@doccategory{}}
\end{DRDC@switch}
\ifx\DRDC@doccat\DRDC@U\else
  \newif\ifDRDC@hascover\DRDC@hascoverfalse
  \newcommand\DRDC@covererror{
    \DRDC@error{Only one of \noexpand\maketitle, \noexpand\makecover
                or \MessageBreak\noexpand\makefrontcover should
                be used}}
  \newcommand\makecover{%
    \ifDRDC@hascover\DRDC@covererror
    \else\DRDC@hascovertrue\DRDC@makecover
    \fi
  }
  \newcommand\maketitle{\makecover}
  \newcommand\makefrontcover{\makecover}
\fi
\renewcommand\thanks{%
  \DRDC@error{\noexpand\thanks not allowed in DRDC class}}
\global\let\@thanks\@empty
\newif\ifDRDC@titletotop \DRDC@titletotopfalse
\newcommand\titletotop{\DRDC@titletotoptrue}
\newcommand\statementstoverso{}%
\newif\ifDRDC@onelineprepared \DRDC@onelinepreparedfalse
\newcommand\onelineprepared{\DRDC@onelinepreparedtrue}
\newlength\preparedskip
\setlength\preparedskip{0.5\baselineskip}
\newlength\infostatementsskip
\setlength\infostatementsskip{8\p@ \@plus\z@ \@minus 2\p@}
\newcommand\titlesize{\huge}
\newcommand\subtitlesize{\large}
\newcommand\authorsize{\small}
\newcommand\preparedsize{\small}
\newcommand\infostatementssize{\scriptsize}
\newcommand\establishmentsize{\LARGE}
\newcommand\docinfosize{\Large}
\newcommand\versostatementssize{\scriptsize}
\newcommand\DRDC@makecover{%
  \DRDC@titlepage
  \DRDC@titleverso
  \DRDC@titlecleanup
}
\newcommand\DRDC@country{CAN}
\newcommand\tclassU{%
  \ifDRDC@isfrench SANS CLASSIFICATION \else UNCLASSIFIED\fi}
\newcommand\DRDC@coverheadfoot{%
  \ifDRDC@isclassified\else
    \oneheadfoot{\DRDC@country~\tclassU}%
    \DRDC@adddraftthisheadfoot
  \fi
}
\newcommand\DRDC@titlepage{%
  \begin{titlepage}%
    \thispagestyle{cover}%
    \DRDC@coverheadfoot
  \let\footnote\relax
  \setcounter{page}{-1}%
  \def\thepage{TitlePage}%
  \renewcommand*\rank[1]{}%
  \sffamily
  \vspace*{45\p@}
  \ifDRDC@titletotop\else\vspace*{0.25in}\fi
  \setbox\DRDC@box\vbox{\parbox{\textwidth}{\raggedright
    {\titlesize\bfseries\@title\par}\vspace*{0.2in}
    {\subtitlesize\slshape
     \ifx\DRDC@subtitle\@empty\else\DRDC@subtitle\fi\par}
    \ifDRDC@titletotop\else\mbox{}\par\fi
  }}%
  \setlength{\DRDC@templen}{44\p@}%
  \addtolength{\DRDC@templen}{-\ht\DRDC@box}%
  \box\DRDC@box
  \ifdim \DRDC@templen > \z@
    \ifDRDC@titletotop
      \vfill
    \else
      \vskip\DRDC@templen
    \fi
  \fi
  {\authorsize\raggedright\DRDC@titleauthors}
  \ifDRDC@titletotop\vspace{-1ex}\fi
  \DRDC@preparedinfo
  \vfill
  \vbox{\infostatementssize
        \setlength\parskip{\infostatementsskip}%
        \DRDC@sensitivitystatements\par
        \DRDC@IPstatements\par
         \ifDRDC@isclassified
        \mbox{\textbf{\U}}\DRDC@termsofrelease\par
        \else
        \ifx\calleddistribution\distribvalue %AV
        \DRDC@termsofrelease\par 
        \else %AV
		\DRDC@limit\relax\par
		\fi %AV
\fi
        \DRDC@CGhandlingstatement\par
        \DRDC@infostatements}
  \vfill
  \@setfontsize\DRDC@CGsize\@ixpt{11}%
  \DRDC@addCGlogo
  \vfill
  \vbox{\parskip 0ex%
        \raggedright
        {\establishmentsize\bfseries\DRDC@agencyname\par}
        \vspace{0.25em}%
        \textbf{\docinfosize\DRDC@doccategory}\par
        \large\DRDC@repnumber\par
        \DRDC@Date
       }%
  \vfill
  \end{titlepage}%
}
\newcommand\DRDC@titlecleanup{
  \global\let\title\relax
  \global\let\author\relax
  \global\let\authors\relax
  \global\let\setdate\relax
  \global\let\and\relax
  \global\let\doccategory\relax
  \global\let\sensitivitystatements\relax
  \global\let\addtosensitivitystatements\relax
  \global\let\DRDC@sensitivitystatements\relax
  \global\let\IPstatements\relax
  \global\let\addtoIPstatements\relax
  \global\let\DRDC@IPstatements\relax
  \global\let\releasestatements\relax
  \global\let\addtoreleasestatements\relax
  \global\let\DRDC@releasestatements\relax
  \global\let\warranties\relax
  \global\let\addtowarranties\relax
  \global\let\DRDC@warranties\relax
  \global\let\disclaimers\relax
  \global\let\addtodisclaimers\relax
  \global\let\DRDC@disclaimers\relax
  \global\let\infostatements\relax
  \global\let\addtoinfostatements\relax
  \global\let\DRDC@infostatements\relax
  \global\let\DRDC@CGadvisorystatement\relax
  \global\let\DRDC@CGhandlingstatement\relax
  \global\let\DRDC@CGnotice\relax
  \global\let\protectIP\relax
  \global\let\admininfo\relax
  \global\let\addtoadmininfo\relax
  \global\let\DRDC@admininfo\relax
  \global\let\standardclauses\relax
  \global\let\addtostandardclauses\relax
  \global\let\DRDC@standardclauses\relax
  \global\let\titlesize\relax
  \global\let\authorsize\relax
  \global\let\preparedsize\relax
  \global\let\infostatementssize\relax
  \global\let\establishmentsize\relax
  \global\let\docinfosize\relax
  \global\let\versostatementssize\relax
  \global\let\DRDC@copyrighttext\relax
  \global\let\originalsignedby\relax
  \global\let\makesignature\relax
  \global\let\repnumber\relax
  \global\let\subtitle\relax
  \global\let\subtitlesize\relax
  \global\let\DRDC@preparedinfo\relax
  \global\let\DRDC@@preparedinfo\relax
  \global\let\DRDC@ECRstatement\relax
  \global\let\DRDC@addECRstatements\relax
  \global\let\clientIP\relax
  \global\let\jointIP\relax
  \global\let\crownIP\relax
  \DRDC@titlepagehook
  \global\let\DRDC@titlepagehook\relax
}
\newcommand\DRDC@titlepagehook{}
\newcommand\DRDC@addCGlogo{}
\newcommand\DRDC@preparedinfo{%
  \ifx\DRDC@client\@empty\else
    \DRDC@@preparedinfo{%
      \DRDC@@preparedline{\DRDC@preparedfortext}{\DRDC@preparedfor}}%
  \fi
}
\newcommand\DRDC@@preparedinfo[1]{%
  \vfill\vbox{\preparedsize\sffamily #1}}
\newcommand\DRDC@@preparedline[2]{
  #1\DRDC@colon
  \ifDRDC@onelineprepared
    {\setlength\DRDC@templen{\hsize}%
     \sbox\DRDC@box{#1\DRDC@colon~}%
     \addtolength\DRDC@templen{-\wd\DRDC@box}
     \parbox[t]{\DRDC@templen}{#2}}
  \else
    \\[\preparedskip]#2
  \fi
}
\ifDRDC@isfrench
\newcommand*\DRDC@Torstext{\textbf{Conditions de diffusion\DRDC@colon} Toute
    distribution ult\'erieure de ce document ou de l'information
    incluse est interdite sans l'approbation \'ecrite de
    \RDDC\ (RDDC).}
\else
\newcommand*\DRDC@Torstext{\textbf{Terms of Release:} Further distribution of
    this document or information contained herein is
    prohibited without the written approval of \DRDC\ (DRDC).}
\fi
\newcommand\calleddistribution{0} %AV
\newcommand\distribvalue{0} %AV
\newcommand\DRDC@limit{\ifx\DRDC@doccat\DRDC@U\DRDC@Torstext\fi}
\newcommand\distribution{%
 \renewcommand\DRDC@limit{\DRDC@Torstext}
 \renewcommand\calleddistribution{1} %AV
}
\ifDRDC@isfrench
  \newcommand\DRDC@infoheader{IMPORTANTS \'ENONC\'ES INFORMATIFS}
\else
  \newcommand\DRDC@infoheader{IMPORTANT INFORMATIVE STATEMENTS}
\fi
\newcommand\DRDC@titleverso{
  \begin{titlepage}
    \DRDC@coverheadfoot
    \sffamily
    \def\thepage{TitlePageBack}%
    \versostatementssize
    \mbox{}%
    \long\def\DRDC@temp{%
      {\bfseries\large\sffamily\DRDC@infoheader}\par
      }%
    \ifx\DRDC@warranties\@empty
      \ifx\DRDC@disclaimers\@empty
        \ifx\DRDC@CGadvisorystatement\@empty
        \else\DRDC@temp\fi
      \else\DRDC@temp\fi
    \else\DRDC@temp\fi

    \DRDC@CGadvisorystatement\par
     \DRDC@warranties\par
      \DRDC@disclaimers\par
      %\DRDC@endorsement\par
    \ifx\DRDC@CGnotice\@empty\else
      \vfill
      \DRDC@CGnotice\par
    \fi
    \ifx\DRDC@admininfo\@empty\else
      \vfill
      \DRDC@admininfo\par
    \fi
    \ifx\DRDC@standardclauses\@empty\else
      \vfill
      \DRDC@standardclauses\par
    \fi
    \vfill
    \parskip1ex
    \raggedright\DRDC@copyrighttext
  \end{titlepage}
}
\setcounter{secnumdepth}{4}
\newcounter{section}
\newcounter{annex}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]
\newcounter{subannex}[annex]
\newcounter{subsubannex}[subannex]
\newcounter{parannex}[subsubannex]
\newcounter{subparannex}[parannex]
\renewcommand\theannex{\@Alph\c@annex}
\ifDRDC@isnumbered
  \renewcommand\thesection      {\@arabic\c@section}
  \renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
  \renewcommand\thesubsubsection{\thesubsection.\@arabic
                                                   \c@subsubsection}
  \renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
  \renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\else
  \let\thesection\relax
  \let\thesubsection\relax
  \let\thesubsubsection\relax
  \let\theparagraph\relax
  \let\thesubparagraph\relax
\fi
\renewcommand\thesubannex{\theannex.\@arabic\c@subannex}
\renewcommand\thesubsubannex{\thesubannex.\@arabic\c@subsubannex}
\renewcommand\theparannex{\thesubsubannex.\@arabic\c@parannex}
\renewcommand\thesubparannex{\theparannex.\@arabic\c@subparannex}
\let\annexmark\@gobble
\let\subannexmark\@gobble
\let\subsubannexmark\@gobble
\let\parannexmark\@gobble
\let\subparannexmark\@gobble
\newcommand\annexname{\appendixname}
\newcommand\annexautorefname{\appendixname}
\AtBeginDocument{\def\appendixautorefname{\appendixname}}
\newcommand\DRDC@startsec[4]{%
  \@startsection{#1}{#2}{\z@}{#3}{0.01ex \@plus \z@}
                {\raggedright\normalfont#4
                 \sffamily\bfseries}%
}
\newlength\DRDC@saveparskip
\newcommand\DRDC@secinit{\DRDC@saveparskip\parskip\parskip\z@}
\newcommand\DRDC@secexit{\par\parskip\DRDC@saveparskip}
\newcommand\DRDC@sectionsize{\LARGE}%
\newcommand\DRDC@subsectionsize{\Large}%
\newcommand\DRDC@subsubsectionsize{\large}%
\newcommand\DRDC@paragraphsize{\normalsize}%
\newcommand\DRDC@subparagraphsize{\normalsize}%
\newcommand\section{\secdef\DRDC@sectionnostar\DRDC@sectionstar}
\newcommand\DRDC@sectionexit{
  \vspace{-8\p@}
  \rule{\textwidth}{0.5\p@}
  \DRDC@secexit
  \nobreak
  \@afterheading
}
\newcommand\DRDC@@startsec{%
  \DRDC@startsec{section}{1}{-3.5ex \@plus \z@ \@minus -.2ex}
                {\DRDC@sectionsize}}
\newcommand\DRDC@sectionstar[1]{
  \DRDC@secinit\DRDC@@startsec*{#1}%
  \addcontentsline{toc}{section}{#1}%
  \DRDC@sectionexit}
\newcounter{DRDC@pages}
\def\DRDC@sectionspecials{
  \ifnum\value{section}<\@ne
    \cleardoublepage
    \setcounter{DRDC@pages}{\value{page}}
    \global\c@page\@ne
    \global\@topnum\z@
  \fi
  \gdef\thepage{\@arabic\c@page}
}
\long\def\DRDC@sectionnostar[#1]#2{%
  \DRDC@secpagebreak
  \DRDC@sectionspecials
  \DRDC@secinit
  \ifDRDC@isnumbered
    \DRDC@@startsec[{#1}]{#2}%
  \else
    \DRDC@@startsec*{#2}%
    \addcontentsline{toc}{section}{#1}%
    \refstepcounter{section}%
  \fi
  \DRDC@sectionexit}
\newif\ifDRDC@inapp \DRDC@inappfalse
\newcommand\annex{\@dblarg{\DRDC@annex}}
\long\def\DRDC@annex[#1]#2{%
   \ifDRDC@inapp\else\appendix\fi
   \cleardoublepage
   \setcounter{equation}{0}%
   \setcounter{figure}{0}%
   \setcounter{table}{0}%
   \DRDC@secinit
   \global\@topnum\z@
   \@ifundefined{Hy@writebookmark}{}{%
     \let\oldHy@writebookmark\Hy@writebookmark
     \def\Hy@writebookmark##1##2{%
        \oldHy@writebookmark{##1}{\appendixname\ ##2}}}
   \def\@seccntformat##1{\appendixname~\csname the##1\endcsname
                      ~~}% \DRDC@colon~}
   \DRDC@startsec{annex}{1}{-3.5ex \@plus \z@ \@minus -.2ex}
                 {\DRDC@sectionsize}[{#1}]{#2}%
   \def\@seccntformat##1{\csname the##1\endcsname\quad}
   \@ifundefined{Hy@writebookmark}{}{%
     \let\Hy@writebookmark\oldHy@writebookmark}%
   \DRDC@sectionexit
}
\newcommand\subsection{%
  \secdef\DRDC@subsectionnostar\DRDC@subsectionstar}
\newcommand\DRDC@startsubsec{%
  \DRDC@startsec{\ifDRDC@inapp subannex\else subsection\fi}{2}
                {-3.25ex\@plus -1ex \@minus -.2ex}
                {\DRDC@subsectionsize}}
\newcommand\DRDC@subsectionstar[1]{%
  \DRDC@secinit\DRDC@startsubsec *{#1}\DRDC@secexit}

\ifDRDC@isnumbered
  \long\def\DRDC@subsectionnostar[#1]#2{%
    \DRDC@secinit\DRDC@startsubsec[{#1}]{#2}\DRDC@secexit}
\else
  \long\def\DRDC@subsectionnostar[#1]#2{%
    \DRDC@subsectionstar{#2}%
    \addcontentsline{toc}{\ifDRDC@inapp subannex%
                          \else subsection\fi}{#1}}%
\fi
\newcommand\subsubsection{%
  \secdef\DRDC@subsubsectionnostar\DRDC@subsubsectionstar}
\newcommand\DRDC@startsubsubsec{%
  \DRDC@startsec{\ifDRDC@inapp subsubannex\else subsubsection\fi}{3}
                {-3.25ex\@plus -1ex \@minus -.2ex}
                {\DRDC@subsubsectionsize}}
\newcommand\DRDC@subsubsectionstar[1]{%
  \DRDC@secinit\DRDC@startsubsubsec *{#1}\DRDC@secexit}

\ifDRDC@isnumbered
  \long\def\DRDC@subsubsectionnostar[#1]#2{%
    \DRDC@secinit\DRDC@startsubsubsec[{#1}]{#2}\DRDC@secexit}
\else
  \long\def\DRDC@subsubsectionnostar[#1]#2{%
    \DRDC@subsubsectionstar{#2}%
    \addcontentsline{toc}{\ifDRDC@inapp subsubannex%
                          \else subsubsection\fi}{#1}}%
\fi
\newcommand\paragraph{\secdef\DRDC@paranostar\DRDC@parastar}
\newcommand\DRDC@startpar{%
  \DRDC@startsec{\ifDRDC@inapp parannex\else paragraph\fi}{4}
                {-3.25ex\@plus -1ex \@minus -.2ex}
                {\DRDC@paragraphsize}}
\newcommand\DRDC@parastar[1]{%
  \DRDC@secinit\DRDC@startpar *{#1}\DRDC@secexit}
\ifDRDC@isnumbered
  \long\def\DRDC@paranostar[#1]#2{%
    \DRDC@secinit\DRDC@startpar[{#1}]{#2}\DRDC@secexit}
\else
  \long\def\DRDC@paranostar[#1]#2{%
    \DRDC@parastar{#2}%
    \addcontentsline{toc}{\ifDRDC@inapp parannex%
                          \else paragraph\fi}{#1}}%
\fi
\newcommand\subparagraph{
  \@startsection{\ifDRDC@inapp subparannex\else subparagraph\fi}
                {5}{\z@}%
                {1.25ex \@plus1ex \@minus .2ex}{-1em}%
                {\raggedright\normalfont\DRDC@subparagraphsize
                 \sffamily\bfseries}}
\newcommand\appendix{
  \ifDRDC@inapp
    \DRDC@error{Annex already defined.}
  \else
    \DRDC@inapptrue
    \DRDC@sectionspecials
    \ifnum\value{section}<\@ne\else\clearpage\fi
    \setcounter{annex}{0}
    \renewcommand\theequation{\theannex.\@arabic\c@equation}
    \renewcommand\thefigure{\theannex.\@arabic\c@figure}
    \renewcommand\thetable{\theannex.\@arabic\c@table}
    \renewcommand\section{\secdef\DRDC@annex\DRDC@sectionstar}
    \newcommand\subannex{\subsection}
    \newcommand\subsubannex{\subsubsection}
    \ifDRDC@ishyperref
       \def\theHequation{\theequation}
       \renewcommand\theHfigure{\thefigure}
       \renewcommand\theHtable{\thetable}
       \newcommand\theHannex{\theannex}
       \newcommand\theHsubannex{\thesubannex}
       \newcommand\theHsubsubannex{\thesubsubannex}
       \newcommand\theHparannex{\theparannex}
       \newcommand\theHsubparannex{\thesubparannex}
    \fi
  \fi
}
\newcounter{DRDC@tempcnt}%
\let\DRDC@oldlist\list
\def\list{%
  \setcounter{DRDC@tempcnt}{\@listdepth}%
  \addtocounter{DRDC@tempcnt}{1}%
  \expandafter\let\expandafter\DRDC@oldlisti
    \csname @list\romannumeral\the\value{DRDC@tempcnt}\endcsname
  \expandafter\def
    \csname @list\romannumeral\the\value{DRDC@tempcnt}\endcsname
    {\DRDC@oldlisti\setlength{\topsep}{\z@ \@plus 2\p@ \@minus 2\p@}}%
    \DRDC@oldlist}
\setlength\leftmargini  {2.5em}
\leftmargin\leftmargini
\setlength\leftmarginii {2.2em}
\setlength\leftmarginiii{1.87em}
\setlength\leftmarginiv {1.7em}
\setlength\leftmarginv  {1em}
\setlength\leftmarginvi {1em}
\setlength\labelsep     {.5em}
\setlength\labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                 \normalfont\bfseries #1}
\def\DRDC@saveandquoteenv#1{%
  \expandafter\def\csname DRDC@save#1\endcsname{}%
  \DRDC@@saveandquoteenv{#1}%
}
\def\DRDC@afterfi#1\fi{\fi#1}
\long\def\DRDC@@saveandquoteenv#1#2\end#3{%
  \def\DRDC@tempa{#1}%
  \def\DRDC@temp{#3}%
  \ifx\DRDC@temp\DRDC@tempa
    \expandafter\g@addto@macro\csname DRDC@save#1\endcsname
      {#2}#2\end{#3}
  \else
    \expandafter\g@addto@macro\csname DRDC@save#1\endcsname
      {#2\end{#3}}#2\end{#3}
    \DRDC@afterfi\DRDC@@saveandquoteenv{#1}%
  \fi
}
\newenvironment{abstract}{%
  \begingroup\ifDRDC@isfrench\selectlanguage{english}\fi
  \section*{\abstractname}\DRDC@saveandquoteenv{abstract}}%
  {\endgroup}
\newenvironment{fabstract}{%
  \begingroup\ifDRDC@isfrench\else\selectlanguage{french}\fi
  \section*{\abstractname}\DRDC@saveandquoteenv{fabstract}}%
  {\endgroup}%
\newcommand*\esignificancename{Significance for defence and security}
\newcommand*\fsignificancename{Importance pour la d\'efense et la
                               s\'ecurit\'e}
\ifDRDC@isfrench
  \newcommand\significancename{\fsignificancename}
\else
  \newcommand\significancename{\esignificancename}
\fi
\ifx\DRDC@doccat\DRDC@U\else
  \newenvironment{significance}
                 {\begingroup
                  \ifDRDC@isfrench\selectlanguage{english}\fi
                  \section*{\significancename}}
                 {\endgroup\clearpage}
  \newenvironment{fsignificance}
                 {\begingroup
                  \ifDRDC@isfrench\else\selectlanguage{french}\fi
                  \section*{\significancename}}
                 {\endgroup\clearpage}
\fi
\newenvironment{acknowledgements}{%
   \clearpage\section*{\acknowledgementname}}{\clearpage}
\newenvironment{titlepage}{
  \newpage
  \thispagestyle{empty}
  \raggedright}{\newpage}
\renewcommand\theequation{\@arabic\c@equation}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\newcounter{figure}
\renewcommand\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
  {\begingroup\DRDC@figpreamble\@float{figure}}
  {\end@float\endgroup}
\newenvironment{figure*}
  {\begingroup\DRDC@figpreamble\@dblfloat{figure}}
  {\end@dblfloat\endgroup}
\newcommand\DRDC@figpreamble{%
  \belowcaptionskip\beyondcaptionskip
  \abovecaptionskip\figtocaptionskip}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
  {\begingroup\DRDC@tablepreamble\@float{table}}
  {\end@float\endgroup}
\newenvironment{table*}
  {\begingroup\DRDC@tablepreamble\@dblfloat{table}}
  {\end@dblfloat\endgroup}
\newcommand\DRDC@tablepreamble{%
  \belowcaptionskip\figtocaptionskip
  \abovecaptionskip\beyondcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@dotsep{4.5}
\newcommand\@tocrmarg{2.55em plus 1fil}
\setcounter{tocdepth}{4}
\newcommand\tableofcontents{%
  \cleardoublepage
  \section*{\contentsname}%
  \@starttoc{toc}%
}
\ifDRDC@isnumbered
  \newlength\DRDC@tocsecbox
  \settowidth\DRDC@tocsecbox{000}
  \newlength\DRDC@tocsubsecbox
  \settowidth\DRDC@tocsubsecbox{000.00}
  \newlength\DRDC@tocsubsubsecbox
  \settowidth\DRDC@tocsubsubsecbox{000.00.0}
  \newlength\DRDC@tocparbox
  \settowidth\DRDC@tocparbox{000.00.0.0}
  \newlength\DRDC@tocsubparbox
  \settowidth\DRDC@tocsubparbox{000.00.0.0.0}
\fi
\newlength\annexlabelwidth
\ifDRDC@isfrench\settowidth\annexlabelwidth{Annexe~M~:~}
\else           \settowidth\annexlabelwidth{Annex~M:~}\fi
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{\normallineskip}%
    \@dottedtocline{1}{\z@skip}{\ifDRDC@isnumbered\DRDC@tocsecbox
                                \else\z@skip\fi}{#1}{#2}
  \fi}
\ifDRDC@isnumbered
  \newcommand*\l@subsection{%
    \@dottedtocline{2}{\DRDC@tocsecbox}{\DRDC@tocsubsecbox}}
  \newcommand*\l@subsubsection{%
    \setlength\DRDC@templen{\DRDC@tocsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocsubsecbox}%
    \@dottedtocline{3}{\DRDC@templen}{\DRDC@tocsubsubsecbox}}
  \newcommand*\l@paragraph{%
    \setlength\DRDC@templen{\DRDC@tocsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocsubsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocsubsubsecbox}%
    \@dottedtocline{4}{\DRDC@templen}{\DRDC@tocparbox}}
  \newcommand*\l@subparagraph{%
    \setlength\DRDC@templen{\DRDC@tocsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocsubsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocsubsubsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocparbox}%
    \@dottedtocline{5}{\DRDC@templen}{\DRDC@tocsubparbox}}
\else
  \newcommand*\l@subsection{%
    \@dottedtocline{2}{0.25in}{\z@skip}}
  \newcommand*\l@subsubsection{%
    \@dottedtocline{3}{0.5in}{\z@skip}}
  \newcommand*\l@paragraph{%
    \@dottedtocline{4}{0.75in}{\z@skip}}
  \newcommand*\l@subparagraph{%
    \@dottedtocline{5}{1.0in}{\z@skip}}
\fi
\newcommand*\l@annex[2]{%
  \ifnum \c@tocdepth >\z@
    \begingroup
    \def\numberline##1{\hb@xt@\@tempdima
                        {\appendixname~##1\DRDC@colon\hfil}}%
    \addpenalty\@secpenalty
    \addvspace{\normallineskip}%
    \@dottedtocline{1}{\z@skip}{\annexlabelwidth}{#1}{#2}
    \endgroup
  \fi}
\AtBeginDocument{
  \ifDRDC@ishyperref
    \newcommand*\toclevel@annex{1}%
    \newcommand*\toclevel@subannex{2}%
    \newcommand*\toclevel@subsubannex{3}%
    \newcommand*\toclevel@parannex{4}%
    \newcommand*\toclevel@subparannex{5}%
  \fi}
\ifDRDC@isnumbered
  \newcommand*\l@subannex{%
    \@dottedtocline{2}{\annexlabelwidth}{\DRDC@tocsubsecbox}}
  \newcommand*\l@subsubannex{%
    \setlength\DRDC@templen{\annexlabelwidth}%
    \addtolength\DRDC@templen{\DRDC@tocsubsecbox}%
    \@dottedtocline{3}{\DRDC@templen}{\DRDC@tocsubsubsecbox}}
  \newcommand*\l@parannex{%
    \setlength\DRDC@templen{\annexlabelwidth}%
    \addtolength\DRDC@templen{\DRDC@tocsubsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocsubsubsecbox}%
    \@dottedtocline{4}{\DRDC@templen}{\DRDC@tocparbox}}
  \newcommand*\l@subparannex{%
    \setlength\DRDC@templen{\annexlabelwidth}%
    \addtolength\DRDC@templen{\DRDC@tocsubsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocsubsubsecbox}%
    \addtolength\DRDC@templen{\DRDC@tocparbox}%
    \@dottedtocline{5}{\DRDC@templen}{\DRDC@tocsubparbox}}
\else
  \newcommand*\l@subannex{%
    \@dottedtocline{2}{\annexlabelwidth}{\z@skip}}
  \newcommand*\l@subsubannex{%
    \setlength\DRDC@templen{\annexlabelwidth}%
    \addtolength\DRDC@templen{0.25in}%
    \@dottedtocline{3}{\DRDC@templen}{\z@skip}}
  \newcommand*\l@parannex{%
    \setlength\DRDC@templen{\annexlabelwidth}%
    \addtolength\DRDC@templen{0.5in}%
    \@dottedtocline{4}{\DRDC@templen}{\z@skip}}
  \newcommand*\l@subparannex{%
    \setlength\DRDC@templen{\annexlabelwidth}%
    \addtolength\DRDC@templen{0.75in}%
    \@dottedtocline{5}{\DRDC@templen}{\z@skip}}
\fi
\newcommand\listoffigures{%
  \section*{\listfigurename}%
  \@starttoc{lof}%
}
\newlength{\figlabelwidth}
\ifDRDC@isfrench
  \settowidth\figlabelwidth{\normalfont Tableau~A.0~:~}
\else
  \settowidth\figlabelwidth{\normalfont Figure~A.0:~}
\fi
\newcommand*\l@figure[2]{%
  \addvspace{\normallineskip}%
  \begingroup
  \def\numberline##1{\hb@xt@\@tempdima
                       {\figurename~##1\DRDC@colon\hfil}}%
  \@dottedtocline{1}{\z@skip}{\figlabelwidth}{#1}{#2}%
  \endgroup}
\newcommand\listoftables{%
  \section*{\listtablename}%
  \@starttoc{lot}%
}
\newcommand*\l@table[2]{%
  \addvspace{\normallineskip}%
  \begingroup
  \def\numberline##1{\hb@xt@\@tempdima
                      {\tablename~##1\DRDC@colon\hfil}}%
  \@dottedtocline{1}{\z@skip}{\figlabelwidth}{#1}{#2}%
  \endgroup}
\newenvironment{thebibliography}[1]{
   \begingroup\raggedright
      \ifx\DRDC@doccat\DRDC@U\else\cleardoublepage\fi
      \section*{\refname}
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist
   \endgroup
}
\newdimen\bibindent
\setlength\bibindent{1.5em}
\let\@openbib@code\@empty
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\renewcommand*{\@biblabel}[1]{[#1]\hfill}
\ifDRDC@authorrefs
  \RequirePackage{natbib}%
  \bibpunct[, ]{(}{)}{,}{a}{}{,}%
  \renewcommand\bibsection{\section*{\refname}\raggedright}%
  \setlength\bibhang{\z@}%
\fi
\newenvironment{theindex}
               {\columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\vspace{-3.5ex}\section*{\indexname}]%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem\raggedright}
               {\onecolumn}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width2.5cm
  \kern2.6\p@}
\newcommand\@makefntext[1]{%
    \parindent\z@
    \noindent
    \@makefnmark\ #1}
\newcommand\DRDC@removeoption[1]{%
  \@expandtwoargs\in@{,#1,}{,\@classoptionslist,}%
  \ifin@
    \def\DRDC@temp##1,#1,##2\@@{\def\@classoptionslist{##1,##2}}%
    \expandafter\DRDC@temp\expandafter,\@classoptionslist,\@@
  \fi
}
\let\DRDC@classoptionslist\@classoptionslist
\DRDC@removeoption{english}
\DRDC@removeoption{french}
\ifDRDC@isfrench
  \RequirePackage[english,frenchb]{babel}[2003/03/18]
  \selectlanguage{french}
  \AtBeginDocument{\selectlanguage{french}}
\else
  \RequirePackage[frenchb,english]{babel}[2003/03/18]
  \selectlanguage{english}
  \AtBeginDocument{\selectlanguage{english}}
\fi
\let\@classoptionslist\DRDC@classoptionslist
\addto{\captionsenglish}{%
  \renewcommand*\significancename{\esignificancename}}
\newcommand*\acknowledgementname{Acknowledgements}
\addto{\captionsenglish}{%
  \renewcommand*\contentsname{Table of contents}%
  \renewcommand*\listfigurename{List of figures}%
  \renewcommand*\listtablename{List of tables}%
  \renewcommand*\tablename{Table}%
  \renewcommand*\appendixname{Annex}%
  \renewcommand*\acknowledgementname{Acknowledgements}%
  \renewcommand*{\establishmentname}{\DRDC@eestablishmentname}%
  \renewcommand*{\DRDC@agencyname}{\DRDC}%
  \renewcommand*\@date{\DRDC@edate}%
  \renewcommand*\DRDC@Date{\DRDC@edate}%
}

\newcommand\DRDC@addtofrench[1]{
  \def\DRDC@temp{\captionsfrenchb}
  \ifx\DRDC@temp\captionsfrench
    \addto{\captionsfrenchb}{#1}%
  \else
    \addto{\captionsfrench}{#1}%
  \fi
}
\DRDC@addtofrench{\renewcommand*\significancename{\fsignificancename}}
\DRDC@addtofrench{
  \renewcommand*\figurename{Figure}%
  \renewcommand*\tablename{Tableau}%
  \renewcommand*\acknowledgementname{Remerciements}%
  \renewcommand*{\establishmentname}{\DRDC@festablishmentname}%
  \renewcommand*{\DRDC@agencyname}{\RDDC}%
  \renewcommand*\@date{\DRDC@fdate}%
  \renewcommand*\DRDC@Date{\DRDC@FDate}%
}
\@ifundefined{frenchbsetup}{%
  \ifDRDC@isfrench
    \def\DRDC@extraf{%
      \bbl@nonfrenchlistspacing
      \ifx\bbl@nonfrenchitems\undefined\bbl@nonfrenchitemize
      \else\bbl@nonfrenchitems
      \fi
      \bbl@nonfrenchindent
    }
    \def\DRDC@temp{\extrasfrenchb}
    \ifx\extrasfrench\DRDC@temp
      \addto{\extrasfrenchb}{\DRDC@extraf}%
    \else
      \addto{\extrasfrench}{\DRDC@extraf}%
    \fi
  \fi
}{%
  \frenchbsetup{IndentFirst=false,ReduceListSpacing=false,%
                CompactItemize=false,StandardItemLabels=false}%
}%
\newcommand\activatepunctuation{%
  \catcode`:\active
  \catcode`;\active
  \catcode`!\active
  \catcode`?\active
}
\newcommand\deactivatepunctuation{%
  \@makeother{:}\@makeother{;}\@makeother{?}\@makeother{!}%
}
{\activatepunctuation
 \gdef\DRDC@colon{:}%
 \gdef\DRDC@semicolon{;}}
\newlength\figtocaptionskip
\newlength\beyondcaptionskip
\setlength\figtocaptionskip{5\p@}
\setlength\beyondcaptionskip{\z@}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{\figtocaptionskip}
\setlength\belowcaptionskip{\beyondcaptionskip}
\newcommand\capfraction{1.0}
\newlength\DRDC@captionsize
\newlength\DRDC@captionmargin
\newcommand\captionflush{\centering}
\long\def\@makecaption#1#2{%
  \def\@makecapnum##1##2{%
    {\normalfont\slshape{\bfseries ##1\DRDC@colon}\ ##2}}%
  \vskip\abovecaptionskip
  \begingroup
    \setlength\DRDC@captionsize{\capfraction\hsize}%
    \setlength\DRDC@captionmargin{0.5\hsize}%
    \addtolength\DRDC@captionmargin{-0.5\DRDC@captionsize}%
    \sbox\DRDC@box{\@makecapnum{#1}{#2}}%
    \ifdim \wd\DRDC@box >\DRDC@captionsize
      {\captionflush
       \advance\leftskip \DRDC@captionmargin
       \advance\rightskip \DRDC@captionmargin
       \@makecapnum{#1}{#2}\par}%
    \else
      \global \@minipagefalse
      \hb@xt@\hsize{\hfil\box\DRDC@box\hfil}%
    \fi
  \endgroup
  \vskip\belowcaptionskip}
\AtBeginDocument{
  \@ifpackageloaded{caption}{%
    \captionsetup{textfont=sl,labelfont={bf,sl},%
                  justification=centering}}{}}
\newcommand*\DRDC@classproperties{%
  \tclassU
  \DRDC@semicolon\ \DRDC@DMCAterm\DRDC@semicolon\ DMC A}
\newcommand\DRDC@addclassproperties{%
  \ifDRDC@ishyperref
    \hypersetup{pdfinfo={Classification={\DRDC@classproperties}}}%
  \fi
  \ifDRDC@pdfa
    \ifpdf
      \newcommand\@pdfclassification{\DRDC@classproperties}%
    \fi
  \fi
}
\ifDRDC@isfrench
  \AtBeginDocument{%
    \def\DRDC@temp{T1}%
    \ifx\DRDC@temp\cf@encoding\DRDC@addclassproperties\fi
  }
\else
  \AtBeginDocument{\DRDC@addclassproperties}
\fi
\setlength\columnsep{10\p@}
\setlength\columnseprule{\z@}
\pagestyle{plain}
\pagenumbering{roman}
\raggedbottom
\frenchspacing
\onecolumn
\endinput
%%
%% End of file `drdc-report.cls'.
