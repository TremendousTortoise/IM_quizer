import csv
import json
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

try:
	import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
	pd = None


APP_ROOT = Path(__file__).resolve().parent
TREATMENTS_CSV = APP_ROOT / "treatments.csv"


@st.cache_data
def load_treatments_data():
	if not TREATMENTS_CSV.exists():
		return None

	if pd is not None:
		return pd.read_csv(
			TREATMENTS_CSV,
			usecols=["title", "section_title", "section_text"],
			low_memory=False,
		)

	rows = []
	with TREATMENTS_CSV.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			rows.append(
				{
					"title": row.get("title"),
					"section_title": row.get("section_title"),
					"section_text": row.get("section_text"),
				}
			)
	return rows


def create_quiz(article, focus, questions=5, lakemedel="", link_text=""):
	prompt_template = """
	-Roll-
	Du är en expert på att skapa utbildningsmaterial och provfrågor av hög kvalitet (t.ex. för läkare eller studenter).

	-Mål-
	Din uppgift är att läsa den bifogade artikeln och skapa ett quiz bestående av {questions} frågor för att testa läsarens förståelse.
	Fokusera på: {focus}

	-Instruktioner-
	1. Analysera artikeln noggrant och utgå från den informationen.
	2. Skapa {questions} flervalsfrågor.
	3. Varje fråga ska ha 4 svarsalternativ (A, B, C, D) där endast ett är korrekt.
	4. Frågestammen får gärna vara utförlig och beskriva ett scenario, en klinisk fallstudie eller en kontext.
	5. Svarsalternativen ska vara koncisa. De felaktiga svaren (distraktorerna) måste vara rimliga och troliga för en okunnig person, men entydigt felaktiga baserat på texten.
	6. Inkludera en utförlig pedagogisk förklaring till varför rätt svar är korrekt.

	-Indata-
	Artikel:
	{article}
	Information om specifika läkemedel:
	{lakemedel}
	Ytterliare information från scrapade länkar:
	{link_text}
	
	-Utdata-
	Returnera enligt följande struktur men formatera i MARKDOWN utan extra förklarande text eller kommentarer.
	Exempel:
	
	## Quiz om "fokuset av quizet"
	
	### Fråga 1
	
	Frågestam här (gärna utförlig och beskrivande)
	
	A. Svarsalternativ A
	B. Svarsalternativ B
	C. Svarsalternativ C
	D. Svarsalternativ D
	
	**Rätt svar: B**
	
	**Förklaring:** Här kommer en pedagogisk förklaring som hjälper läsaren att förstå varför det rätta svaret är korrekt och varför de andra alternativen är felaktiga.
	
	---
	
	### Fråga 2
	
	och så vidare...
	"""
	prompt = ChatPromptTemplate.from_template(prompt_template)
	inputs = {
		"article": article,
		"questions": questions,
		"focus": focus,
		"lakemedel": lakemedel,
		"link_text": link_text,
	}

	api_key = st.secrets.get("JOHAN_AZURE_OPENAI_KEY")
	api_version = st.secrets.get("JOHAN_AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
	api_endpoint = st.secrets.get("JOHAN_AZURE_OPENAI_ENDPOINT")
	deployment = st.secrets.get("JOHAN_AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat")

	if not api_key or not api_endpoint:
		raise RuntimeError("Azure OpenAI secrets are missing.")

	llm = AzureChatOpenAI(
		azure_deployment=deployment,
		api_version=api_version,
		azure_endpoint=api_endpoint,
		api_key=api_key,
		temperature=0.0,
	)

	chain = prompt | llm | StrOutputParser()
	return chain.invoke(inputs)


@st.cache_data
def scrape_links(links):
	results = []
	headers = {
		"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
		"(KHTML, like Gecko) Chrome/121.0 Safari/537.36",
	}

	for link in links:
		try:
			response = requests.get(link, headers=headers, timeout=20)
			response.raise_for_status()
			soup = BeautifulSoup(response.text, "html.parser")
			for tag in soup(["script", "style", "noscript"]):
				tag.decompose()
			text = " ".join(soup.get_text(separator=" ").split())
			results.append({"url": link, "text": text})
		except Exception as exc:  # noqa: BLE001 - surface errors in UI
			results.append({"url": link, "error": str(exc)})

	return results


st.set_page_config(page_title="IM quizzer", layout="wide")
col1, col2, col3 = st.columns([2,1,2])
with col2:
    st.image("im-logo-white-text.png", width="stretch")
st.title("IM Quizzer")
st.divider()
st.markdown("<h4>Välj innehåll och skapa quiz:</h4>", unsafe_allow_html=True)

links_text = st.text_area("Länkar att skrapa, en per rad (valfritt)", placeholder="https://fass.se/exempel/\nhttps://example.com/")
prompt_text = st.text_area("Prompt", placeholder="Vad ska quizet fokusera på? Ex biverkningar, interaktioner mm.")
questions = st.slider("Antal frågor", min_value=1, max_value=10, value=5, step=1)
lakemedel_text = st.text_area("Information om specifika läkemedel (valfritt)")

st.divider()


st.subheader("Välj innehåll från internetmedicin")
data = load_treatments_data()

selected_titles = []
selected_sections = []
section_text_output = []

if data is None:
	st.error("internetmedicin data kunde inte hittas i projektmappen.")
else:
	if pd is not None and hasattr(data, "columns"):
		titles = sorted(data["title"].dropna().unique())
		selected_titles = st.multiselect("Titel", options=titles)
		filtered = data[data["title"].isin(selected_titles)]
		filtered = filtered.dropna(subset=["title", "section_title"])
		filtered = filtered.assign(
			section_label=filtered["title"].astype(str).str.cat(
				filtered["section_title"].astype(str), sep=" - "
			)
		)
		section_labels = sorted(filtered["section_label"].unique())
		selected_sections = st.multiselect(
			"Avsnitt",
			options=section_labels,
		)

		if selected_sections:
			selected_rows = filtered[filtered["section_label"].isin(selected_sections)]
			for section_label in selected_sections:
				texts = (
					selected_rows[selected_rows["section_label"] == section_label][
						"section_text"
					]
					.dropna()
				)
				for text in texts:
					section_text_output.append(str(text))
		else:
			st.info("Välj ett eller flera avsnitt för att visa text.")
	else:
		titles = sorted({row["title"] for row in data if row.get("title")})
		selected_titles = st.multiselect("Titel", options=titles)
		filtered = [row for row in data if row.get("title") in selected_titles]
		section_labels = sorted(
			{
				f"{row.get('title')} - {row.get('section_title')}"
				for row in filtered
				if row.get("title") and row.get("section_title")
			}
		)
		selected_sections = st.multiselect(
			"Avsnitt",
			options=section_labels,
		)

		if selected_sections:
			for section_label in selected_sections:
				texts = [
					row.get("section_text")
					for row in filtered
					if (
						row.get("section_text")
						and f"{row.get('title')} - {row.get('section_title')}" == section_label
					)
				]
				for text in texts:
					section_text_output.append(str(text))
		else:
			st.info("Välj ett eller flera avsnitt för att visa text.")

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
	st.markdown("<h1 style='text-align: center;'>Skapa Quiz</h1>", unsafe_allow_html=True)
	generate_quiz = st.button("Skapa quiz", width="stretch")


if generate_quiz:
	article_text = "\n\n".join(section_text_output).strip()
	links = [link.strip() for link in links_text.splitlines() if link.strip()]

	if links:
		with st.spinner("Skrapar länkar..."):
			scraped = scrape_links(links)
			scraped_texts = [item["text"] for item in scraped if "text" in item]
			article_text = "\n\n".join([article_text] + scraped_texts).strip()
			scrape_errors = [item for item in scraped if "error" in item]
			if scrape_errors:
				st.warning("Vissa länkar kunde inte skrapas.")

	if not article_text:
		st.error("Välj minst ett avsnitt för att skapa quiz.")
	else:
		try:
			with st.spinner("Skapar quiz..."):
				quiz_json = create_quiz(
					article=article_text,
					focus=prompt_text or "allmän förståelse",
					questions=questions,
					lakemedel=lakemedel_text,
					link_text=scraped_texts if links else "",
				)
			st.divider()
			st.markdown(quiz_json)
		except Exception as exc:  # noqa: BLE001 - show config/runtime errors
			st.error(str(exc))
