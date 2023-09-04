from typing import List

from pydantic import parse_obj_as
from qdrant_client import QdrantClient

from streaming_pipeline import initialize
from streaming_pipeline.embeddings import EmbeddingModelSingleton
from streaming_pipeline.models import NewsArticle
from streaming_pipeline.qdrant import QdrantVectorOutput

mocked_news = [
    [
        {
            "T": "n",
            "id": 32505835,
            "headline": "Sangamo Therapeutics Receives U.S. FDA Fast Track Designation For Isaralgagene Civaparvovec For The Treatment Of Fabry Disease", # noqa: E501
            "summary": "Sangamo Therapeutics, Inc. (NASDAQ:SGMO), a genomic medicine company, today announced that the U.S. Food and Drug Administration (FDA) has granted Fast Track Designation to isaralgagene civaparvovec, or ST-920, a wholly", # noqa: E501
            "author": "Benzinga Newsdesk",
            "created_at": "2023-05-22T12:06:20Z",
            "updated_at": "2023-05-22T12:06:21Z",
            "url": "https://www.benzinga.com/general/biotech/23/05/32505835/sangamo-therapeutics-receives-u-s-fda-fast-track-designation-for-isaralgagene-civaparvovec-for-th",
            "content": '\u003cp\u003eSangamo Therapeutics, Inc. (NASDAQ:\u003ca class="ticker" href="https://www.benzinga.com/stock/SGMO#NASDAQ"\u003eSGMO\u003c/a\u003e), a genomic medicine company, today announced that the U.S. Food and Drug Administration (FDA) has granted Fast Track Designation to isaralgagene civaparvovec, or ST-920, a wholly owned gene therapy product candidate for the treatment of Fabry disease.\u003c/p\u003e\u003cp\u003eFast Track designation aims to facilitate the development and expedite the review of new therapeutics that are intended to treat serious or life-threatening conditions and that demonstrate the potential to address unmet medical needs. Companies granted this designation are given the opportunity for more frequent interactions with the FDA. These clinical programs may also be eligible to apply for Accelerated Approval and Priority Review if relevant criteria are met. The FDA has previously granted ST-920 Orphan Drug Designation.\u003c/p\u003e\u003cp\u003e"We are thrilled with the FDA\'s decision to grant Fast Track Designation for ST-920. Fabry is a debilitating disease with life-long impact," said Nathalie Dubois-Stringfellow, Ph.D, Sangamo\'s Senior Vice President, Chief Development Officer. "This decision from the FDA underscores the potential for ST-920 to address a serious unmet need and serve as a meaningful therapeutic option for patients with Fabry disease. We are highly encouraged by this promising development and look forward to our expected meeting with the FDA on Phase 3 trial design in the summer."\u003c/p\u003e\u003cp\u003eST-920 is currently being evaluated in the Phase 1/2 STAAR study, with a total of 20 patients dosed to date. In February 2023, Sangamo announced promising results from the STAAR study via an oral presentation at the 19th Annual WORLD\u003ci\u003eSymposium, \u003c/i\u003eshowing sustained, elevated expression of alpha-galactosidase A (α-Gal A) activity in the 13 dosed patients as of the data cutoff, 78% globotriaosylceramide (Gb3) substrate clearance at 6-months and 77% reduction in urine podocyte loss in one of the first kidney biopsies, and a clinically meaningful and statistically significant increase in mean general health scores, as measured by the SF-36 General Health survey. A copy of the presentation is available in the \u003ca href="https://cts.businesswire.com/ct/CT?id=smartlink\u0026amp;url=https%3A%2F%2Finvestor.sangamo.com%2Fpresentations\u0026amp;esheet=53403648\u0026amp;newsitemid=20230522005108\u0026amp;lan=en-US\u0026amp;anchor=Presentations+section+of+the+Sangamo+website\u0026amp;index=1\u0026amp;md5=b8ed12a9ebf94820f3dec0b898757f74"\u003ePresentations section of the Sangamo website\u003c/a\u003e. Sangamo is currently preparing for a potential Phase 3 trial and plans to meet with the FDA on the proposed Phase 3 study design in the summer, with a trial start anticipated by the end of 2023, depending on regulatory interactions.\u003c/p\u003e\u003cp\u003e\u003cstrong\u003eAbout the STAAR Study\u003c/strong\u003e\u003c/p\u003e\u003cp\u003eThe Phase 1/2 STAAR study is a global open-label, single-dose, dose-ranging, multicenter clinical study designed to evaluate the safety and tolerability of isaralgagene civaparvovec, or ST-920, a gene therapy product candidate in patients with Fabry disease. Isaralgagene civaparvovec requires a one-time infusion without preconditioning. The STAAR study is enrolling patients who are on ERT, are ERT pseudo-naïve (defined as having been off ERT for six or more months), or who are ERT-naïve. The U.S. Food and Drug Administration has granted Orphan Drug and Fast Track designation to isaralgagene civaparvovec, which has also received Orphan Medicinal Product designation from the European Medicines Agency.\u003c/p\u003e', # noqa: E501
            "symbols": ["SGMO"],
            "source": "benzinga",
        }
    ],
    [
        {
            "T": "n",
            "id": 32052192,
            "headline": "What\u0026#39;s Going On With ContraFecta Stock Today",
            "summary": "\n\tContraFect Corporation (NASDAQ: CFRX) shares are trading higher Thursday morning. However, there is no specific news to justify the move.\n", # noqa: E501
            "author": "Vandana Singh",
            "created_at": "2023-04-27T18:24:49Z",
            "updated_at": "2023-04-27T18:24:49Z",
            "url": "https://www.benzinga.com/general/biotech/23/04/32052192/whats-going-on-with-contrafecta-stock-today",
            "content": '\u003cul\u003e\r\n\t\u003cli\u003e\u003cstrong\u003eContraFect Corporation\u0026nbsp;\u003c/strong\u003e(NASDAQ:\u003ca class="ticker" href="https://www.benzinga.com/stock/CFRX#NASDAQ"\u003eCFRX\u003c/a\u003e) shares are trading higher Thursday morning. However, there is no specific news to justify the move.\u003c/li\u003e\r\n\t\u003cli\u003eWednesday morning, ContraFect\u0026nbsp;\u003ca class="editor-rtfLink" href="https://www.benzinga.com/pressreleases/23/04/g32009856/contrafect-announces-first-patient-dosed-in-the-phase-1b2-study-of-exebacase-in-patients-with-chro" style="color:#4a6ee0; background:transparent; margin-top:0pt; margin-bottom:0pt" target="_blank"\u003eannounced the dosing\u003c/a\u003e\u0026nbsp;of the first patient in Phase 1b/2 of exebacase in the setting of an arthroscopic debridement, antibiotics, irrigation, and retention procedure in patients with chronic prosthetic joint infections of the knee due to\u0026nbsp;\u003cem\u003eStaphylococcus aureus\u003c/em\u003e\u0026nbsp;or Coagulase-Negative Staphylococci.\u003c/li\u003e\r\n\t\u003cli\u003eThe study was initiated\u0026nbsp;\u003ca class="editor-rtfLink" href="https://www.benzinga.com/pressreleases/23/04/g31631700/contrafect-announces-initiation-of-a-phase-1b2-study-of-exebacase-in-patients-with-chronic-prosthe" style="color:#4a6ee0; background:transparent; margin-top:0pt; margin-bottom:0pt" target="_blank"\u003eearlier this month\u003c/a\u003e.\u003c/li\u003e\r\n\t\u003cli\u003eContraFect stock is gaining on heavy volume, with a session volume of 55 million shares traded, compared to the trailing 100-day volume of 3.08 million shares.\u003c/li\u003e\r\n\t\u003cli\u003eAccording to data from\u0026nbsp;\u003ca class="editor-rtfLink" href="https://benzinga.grsm.io/register174" style="color:#4a6ee0; background:transparent; margin-top:0pt; margin-bottom:0pt" target="_blank"\u003eBenzinga Pro\u003c/a\u003e, CFRX has a 52-week high of $362.4 and a 52-week low of $0.90.\u003c/li\u003e\r\n\t\u003cli\u003e\u003cstrong\u003ePrice Action:\u003c/strong\u003e\u0026nbsp;CFRX shares are up 68.20% at $2.22 on the last check Thursday.\u003c/li\u003e\r\n\u003c/ul\u003e\r\n ', # noqa: E501
            "symbols": ["CFRX"],
            "source": "benzinga",
        }
    ],
]

if __name__ == "__main__":
    initialize()

    model = EmbeddingModelSingleton()

    for articles in mocked_news:
        articles = parse_obj_as(List[NewsArticle], articles)
        for article in articles:
            document = article.to_document()
            document = document.compute_chunks(model)
            document = document.compute_embeddings(model)

            print("-" * 100)
            print("Document: ")
            print()
            print(document)
            print("-" * 100)
            print()

            output = QdrantVectorOutput(
                vector_size=model.max_input_length, client=QdrantClient(":memory:")
            )
            output_sink = output.build(1, 1)
            output_sink.write(document)
