import requests
from urllib.parse import quote
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from datetime import datetime


_ = load_dotenv()


def get_full_text(url, word_limit=1000):
    """Retrieve text from webpages"""
    try:
        page_response = requests.get(url)
        if page_response.status_code != 200:
            raise ConnectionError(
                f"Failed to retrieve the page content with status code: {page_response.status_code}"
            )

        soup = BeautifulSoup(page_response.content, "html.parser")
        paragraphs = soup.find_all("p")
        full_text = " ".join([p.get_text() for p in paragraphs])
        words = full_text.split()
        text = " ".join(words[:word_limit]) if len(words) > word_limit else full_text

        return text

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Error retrieving full text from page: {e}")


def google_search(query, k=1):
    google_search_api = os.getenv("GOOGLE_SEARCH")
    cx = os.getenv("cx")

    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"q": query, "key": google_search_api, "cx": cx},
        )

        if response.status_code != 200:
            raise ConnectionError(
                f"Google Search API failed with status code: {response.status_code}"
            )

        result_summary = []
        data = response.json()
        for i in range(k):
            if "items" in data and data["items"]:
                result = data["items"][i]
                result_summary.append(
                    {"title": result.get("title"), "link": result.get("link")}
                )

        return result_summary

    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Google Search API request failed: {e}")


def scrape_wiki(query, word_limit=1000):
    base_url = "https://en.wikipedia.org/wiki/"
    page_url = f"{base_url}{quote(query)}"

    try:
        response = requests.get(page_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p", limit=5)
        content = " ".join(
            [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
        )
        words = content.split()
        limited_content = (
            " ".join(words[:word_limit]) if len(words) > word_limit else content
        )

        return [{"title": query, "link": page_url, "content": limited_content}]

    except requests.exceptions.RequestException as _:
        # force wiki search with gsearch API
        query = query + " Wikipedia"
        result_wiki = google_search(query)
        text = get_full_text(result_wiki[0]["link"])
        return [{"title": query, "link": result_wiki[0]["link"], "content": text}]


def scrape_stackexchange(query, k=1):
    stackexchange_api_url = "https://api.stackexchange.com/2.3"
    sites = ["privacy policies", "law", "legal laws", "policies"]
    questions_data = []

    for site in sites:
        # Build API query
        params = {
            "site": site,
            "tagged": "privacy",
            "q": "",
            "sort": "votes",
            "order": "desc",
        }

        try:
            response = requests.get(
                f"{stackexchange_api_url}/search/advanced", params=params
            )
            if response.status_code == 200:
                for question in response.json()["items"]:
                    questions_data.append(
                        {
                            "site": site,
                            "title": question["title"],
                            "link": question["link"],
                            "score": question["score"],
                        }
                    )
        except Exception:
            print("StackEx error")

    return questions_data


def scrape_CFPB(query, k):
    params = {
        "company": "pathway",
        "issue": "policies",
        "date_received_max": datetime.now().strftime("%Y-%m-%d"),
    }

    CFPB_api_url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

    complaints_data = []
    response = requests.get(CFPB_api_url, params=params)

    if response.status_code == 200:
        for complaint in response.json()["hits"]["hits"]:
            complaint_data = complaint["_source"]
            complaints_data.append(
                {
                    "title": complaint_data.get("issue", ""),
                    "content": complaint_data.get("state", ""),
                }
            )

    return complaints_data


def scrape_reddit(query, k=1):
    relevant_subreddits = ["privacy", "PrivacyGuides", "legal"]
    url = "https://socialgrep.p.rapidapi.com/search/posts"

    posts_data = []
    for subreddit_name in relevant_subreddits:
        query_form = f"/r/{subreddit_name},{query}"
        querystring = {"query": query_form}

        headers = {
            "x-rapidapi-key": os.getenv("x_rapidapi_key"),
            "x-rapidapi-host": "socialgrep.p.rapidapi.com",
        }

        response = requests.get(url, headers=headers, params=querystring)
        try:
            posts_data.append(response.json())
        except Exception as e:
            print("Reddit API failed with error:", e)

    return posts_data


def search(query, k=1):
    methods = [google_search, scrape_wiki, scrape_CFPB]
    current_method = 0
    result = []
    for _ in range(len(methods)):
        try:
            result.append(methods[current_method](query, k))
            print(f"Using {methods[current_method].__name__}")
        except ConnectionError as e:
            print(e)
            result.append({})
        current_method = (current_method + 1) % len(methods)
    return result
