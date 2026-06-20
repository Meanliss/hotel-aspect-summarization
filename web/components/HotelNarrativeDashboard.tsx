"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ASPECT_LABEL,
  METHOD_IDS,
  anonymizePropertyText,
  loadSpaceData,
  type MethodId,
  type SpaceAspect,
  type SpaceData,
  type SpaceEntity,
} from "@/lib/space";

const ASPECT_VISUALS: Record<
  SpaceAspect,
  { mark: string; tag: string; tone: string; gradient: string; image?: string }
> = {
  building: {
    mark: "B",
    tag: "High reliability",
    tone: "text-[var(--primary)] bg-[var(--primary-fixed)]/55",
    gradient: "from-[#2f251c] via-[#6f573d] to-[#c2a472]",
    image:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuD4hTbGIUbSJ2ZdGYW3YuhckxNHJMCDe4KnuVHDOnEDnzTWdNlcohSviEsBkOWcZSJgospbndPtJxibXINXcn3gwvmTqKPfiO8cyQ_bcfzxD7oOfIBgct1gfVEwn4dMFqv-VBgCAIm8OWEQl5_6awncvAe2qxYuGh6SmQKalcZwhIJ1h_phrLgISjvMbq9Tgy2p58LUSBScuwYH8aYfowHzRXKA3A4ymfItDflwFJE847niy24gpxaI7ak5-CISHUjXfYHvydJeUIg",
  },
  cleanliness: {
    mark: "C",
    tag: "Distinct signal",
    tone: "text-[var(--secondary)] bg-[var(--secondary-fixed)]/55",
    gradient: "from-[#4c3b2b] via-[#b9874a] to-[#f1ddbb]",
  },
  food: {
    mark: "F",
    tag: "Experience marker",
    tone: "text-[var(--secondary)] bg-[var(--secondary-fixed)]/55",
    gradient: "from-[#3f3328] via-[#9b6f3f] to-[#ead2a6]",
    image:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuAsr81m4xTolgYwroi-XIuj5pLCh0TMYn1D59C3ZE80bOfbdzTO9nN4VsaubHB8bv8BXG-HvtLXT7TdK-NsEbWHA59ehP6_8VJwbSZK6HvXXvWqRl0KAaUyFieWzb6ylN3lN9EBYq6JsdplxhzogQu8_-v6xOK9780-aHHZaJse_RX13v5sHukz-VohRrxaOJzux_TNAG1z6yl6JQnqDjaocbbAlrvJDMA6ZJtlq7WDrgxbhSUYjuOflh7uhgB1Lu0gNQ6ik2_53B8",
  },
  location: {
    mark: "L",
    tag: "Strong retention",
    tone: "text-[var(--tertiary)] bg-[var(--tertiary-fixed)]/65",
    gradient: "from-[#283525] via-[#6b7653] to-[#d8c08d]",
    image:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuCBTaNApxxvVM55eWyoGunIbhK5Kwqv0ldpA2jHXS-cbewZ5U1fcM_qYCB_PwcaLmPB5q8ZCd-uwhxFN4JQJm-n537zUehK35a3Lu-0c5_GLGDSgtZuJ5nIQkyy4cE6xJelgkFD0ZcNo3xCOxrRzalu4-ANJSe1FScl9IsuWSDL2oYkOkXQUVoDvibEOgX1mi_PzKzqPWVKXYqW5P1KIToKFICdKXtTy3s4i_3jmw2-S7IalVwvTibHHS6Va3hiCKKlvFn1jqdVcY8",
  },
  rooms: {
    mark: "R",
    tag: "Guest comfort",
    tone: "text-[var(--primary)] bg-[var(--primary-fixed)]/55",
    gradient: "from-[#2c241d] via-[#7c674e] to-[#d1b485]",
    image:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuAEBUPlTMByYEYpyI83i03jrgIPTjQ_KymGOHEaYoiBTprmP95H0PYRICZkNzsyg3iCAbN4EsQsdtVvBS78pUu5fh1DRw-Gii5ao2msJ-P2QyphXlqh3mll4L8vWYvekKlRBq0aXc4LE9cO1meUoGWJaeZ55Y_CPbgPl7KaB10IIFO3_uhmn0JPzbTXSOweoPU0ZSvJEqkj5GBV9Qf2PnpD5vdoB1mHM4FqIqiqjiABG_BtRXma6HZymCgeX-ga7tCZT4cQ-ylkqKY",
  },
  service: {
    mark: "S",
    tag: "Precision",
    tone: "text-[var(--tertiary)] bg-[var(--tertiary-fixed)]/65",
    gradient: "from-[#243027] via-[#55684f] to-[#d7c4a2]",
    image:
      "https://lh3.googleusercontent.com/aida-public/AB6AXuBuiFuKfr1mkfoSewQGZVFq3eyh81uZhpmzsIYVoDS8H2Ogh7E6n9EqnoU4uyWJsiM1G7YK3nmQTSjTlZi670eHpDJbZW5GA2inr4fn-Nsn6_CyHWX3X7UfjR0yr2gQ3KNN2BsYCE7qQ_xkUKXhHNcGS0pAr-E_Ar4Dn_c6yXziIdFsaMFQj5AEl2_cfofGgqp7jNdnPb6jD4Nda5kngC6aSP3fxnYT7oFsvnsA4inzxQ9hfiWB2TG-WK2keIYUI-UhqosDXFj44to",
  },
};

const METHOD_LABEL: Record<MethodId, string> = {
  m1: "M1 Extractive",
  m2: "M2 Abstractive",
  m3: "M3 Keyword Split",
  m4: "M4 BERT-ABSA",
};

function getSummary(entity: SpaceEntity, method: MethodId, aspect: SpaceAspect) {
  const cell = entity.methods[method]?.aspects?.[aspect];
  if (!cell) return "No generated summary for this aspect.";
  if (cell.overall) return anonymizePropertyText(cell.overall, entity.entity_name);
  const text =
    [
      cell.positive ? `Positive: ${cell.positive}` : "",
      cell.negative ? `Negative: ${cell.negative}` : "",
    ]
      .filter(Boolean)
      .join(" ") || "No generated summary for this aspect.";
  return anonymizePropertyText(text, entity.entity_name);
}

function scoreFor(entity: SpaceEntity, method: MethodId) {
  const aspects = Object.values(entity.methods[method]?.aspects ?? {});
  const filled = aspects.filter(
    (cell) => cell.overall || cell.positive || cell.negative,
  ).length;
  return Math.min(9.8, 7.4 + filled * 0.32).toFixed(1);
}

function sampleLabel(index: number) {
  return `Property Sample ${String(index + 1).padStart(2, "0")}`;
}

export function HotelNarrativeDashboard({
  initialData,
}: {
  initialData?: SpaceData;
}) {
  const initialEntityId = initialData?.entities[0]?.entity_id ?? "";
  const [data, setData] = useState<SpaceData | null>(initialData ?? null);
  const [error, setError] = useState("");
  const [entityId, setEntityId] = useState(initialEntityId);
  const [method, setMethod] = useState<MethodId>("m4");

  useEffect(() => {
    if (initialData) return;
    let cancelled = false;
    loadSpaceData()
      .then((payload) => {
        if (!cancelled) {
          setData(payload);
          setEntityId(payload.entities[0]?.entity_id ?? "");
        }
      })
      .catch((reason) => {
        if (!cancelled) setError(String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, [initialData]);

  const entity = useMemo(
    () =>
      data?.entities.find((item) => item.entity_id === entityId) ??
      data?.entities[0],
    [data, entityId],
  );

  if (error) {
    return (
      <div className="rounded-lg bg-rose-50 p-4 text-sm text-rose-700 ring-1 ring-rose-200">
        {error}
      </div>
    );
  }
  if (!data || !entity) {
    return (
      <div className="text-sm text-[var(--on-surface-variant)]">
        Loading dashboard...
      </div>
    );
  }

  const score = scoreFor(entity, method);
  const aspects = data.aspects as SpaceAspect[];
  const entityIndex = Math.max(
    0,
    data.entities.findIndex((item) => item.entity_id === entity.entity_id),
  );
  const activeSampleLabel = sampleLabel(entityIndex);

  return (
    <div className="space-y-16">
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-4 shadow-[var(--shadow-soft)]">
        <div className="flex flex-wrap gap-4">
          <label className="flex items-center gap-2 rounded-lg border border-[var(--outline-variant)] bg-[var(--surface-container-low)] px-4 py-2">
            <span className="font-body text-xs font-semibold uppercase text-[var(--on-surface-variant)]">
              Sample:
            </span>
            <select
              value={entity.entity_id}
              onChange={(event) => setEntityId(event.target.value)}
              className="border-none bg-transparent p-0 font-body text-sm font-semibold text-[var(--primary)] focus:ring-0"
            >
              {data.entities.map((item, index) => (
                <option key={item.entity_id} value={item.entity_id}>
                  {sampleLabel(index)} - {item.split}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 rounded-lg border border-[var(--outline-variant)] bg-[var(--surface-container-low)] px-4 py-2">
            <span className="font-body text-xs font-semibold uppercase text-[var(--on-surface-variant)]">
              Method:
            </span>
            <select
              value={method}
              onChange={(event) => setMethod(event.target.value as MethodId)}
              className="border-none bg-transparent p-0 font-body text-sm font-semibold text-[var(--primary)] focus:ring-0"
            >
              {METHOD_IDS.map((item) => (
                <option key={item} value={item}>
                  {METHOD_LABEL[item]}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="flex items-center gap-2 text-[var(--on-surface-variant)]">
          <span className="material-symbols-outlined text-[20px]">sort</span>
          <span className="font-body text-xs font-semibold uppercase tracking-[0.1em]">
            Sorted by narrative relevance
          </span>
        </div>
      </div>

      <section>
        <div className="mb-12 flex flex-col gap-5 border-b-2 border-[rgba(52,38,27,0.12)] pb-4 md:flex-row md:items-end md:justify-between">
          <div>
            <span className="mb-2 inline-block rounded-full bg-[var(--primary)] px-3 py-1 font-body text-xs font-semibold uppercase tracking-[0.08em] text-white">
              Portfolio Focus
            </span>
            <h2 className="font-headline text-3xl font-bold text-[var(--primary)]">
              {activeSampleLabel}
            </h2>
            <p className="mt-1 font-body text-sm text-[var(--on-surface-variant)]">
              Anonymous benchmark group - {entity.split} - {METHOD_LABEL[method]}
            </p>
          </div>
          <div className="text-left md:text-right">
            <span className="font-headline text-5xl font-bold leading-none text-[var(--primary)]">
              {score}
            </span>
            <div className="font-body text-xs font-semibold uppercase tracking-[0.18em] text-[var(--on-surface-variant)]">
              Portfolio score
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-12">
          {aspects.map((aspect, index) => {
            const visual = ASPECT_VISUALS[aspect];
            const reversed = index % 2 === 1;
            return (
              <article
                key={aspect}
                className={`flex flex-col items-center gap-8 md:gap-16 ${
                  reversed ? "md:flex-row-reverse" : "md:flex-row"
                }`}
              >
                <div
                  aria-label={`${ASPECT_LABEL[aspect] ?? aspect} visual`}
                  className={`relative h-64 w-full overflow-hidden rounded-2xl bg-gradient-to-br ${visual.gradient} shadow-[var(--shadow-soft)] md:h-80 md:w-1/2`}
                >
                  {visual.image ? (
                    <img
                      alt={`${ASPECT_LABEL[aspect] ?? aspect} visual`}
                      className="h-full w-full object-cover transition duration-700 hover:scale-[1.03]"
                      src={visual.image}
                    />
                  ) : (
                    <>
                      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(255,255,255,0.24),transparent_22rem)]" />
                      <div className="absolute -bottom-20 -right-14 h-56 w-56 rounded-full border border-white/20 bg-white/10" />
                      <div className="absolute left-8 top-8 h-24 w-24 rounded-full border border-white/20 bg-black/10" />
                      <div className="absolute inset-x-8 bottom-8 flex items-end justify-between border-t border-white/20 pt-5 text-white/85">
                        <span className="font-headline text-5xl font-bold">
                          {visual.mark}
                        </span>
                        <span className="font-body text-xs font-semibold uppercase tracking-[0.22em]">
                          Anonymous signal
                        </span>
                      </div>
                    </>
                  )}
                </div>
                <div className="flex w-full flex-col gap-4 md:w-1/2">
                  <div className="flex items-center gap-3 text-[var(--primary)]">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--primary-fixed)] font-headline text-sm font-bold text-[var(--primary)]">
                      {visual.mark}
                    </span>
                    <h3 className="font-headline text-xl font-semibold uppercase tracking-tight text-[var(--primary)]">
                      {ASPECT_LABEL[aspect] ?? aspect}
                    </h3>
                  </div>
                  <p className="font-body text-lg leading-relaxed text-[var(--on-surface-variant)]">
                    {getSummary(entity, method, aspect)}
                  </p>
                  <div
                    className={`self-start rounded px-3 py-1 font-body text-xs font-bold uppercase tracking-[0.08em] ${visual.tone}`}
                  >
                    {visual.tag}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      </section>

      <section className="overflow-hidden rounded-2xl bg-[var(--primary)] text-white shadow-[var(--shadow-soft)]">
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px]">
          <div className="p-8 md:p-10">
            <div className="mb-3 font-body text-xs font-semibold uppercase tracking-[0.18em] text-[var(--secondary-fixed)]">
              Deep Insight Report
            </div>
            <h2 className="font-headline text-3xl font-bold">
              From raw reviews to portfolio narrative
            </h2>
            <p className="mt-4 max-w-2xl font-body text-base leading-relaxed text-[#efe6d6]">
              This dashboard presents anonymized hospitality review summaries for
              a portfolio-level draft. Use Trace to audit the sentence-level
              evidence behind each method without foregrounding a single property
              name.
            </p>
          </div>
          <div className="bg-white/10 p-8 md:p-10">
            <div className="font-headline text-4xl font-bold">
              {aspects.length}
            </div>
            <div className="mt-1 font-body text-xs font-semibold uppercase tracking-[0.16em] text-[#efe6d6]">
              Aspect dimensions
            </div>
            <div className="mt-6 font-headline text-4xl font-bold">
              {METHOD_IDS.length}
            </div>
            <div className="mt-1 font-body text-xs font-semibold uppercase tracking-[0.16em] text-[#efe6d6]">
              Pipeline methods
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
