import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def search_pubmed(query: str, max_results: int = 2):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        # 1. Search PubMed for PMIDs
        search_params = urllib.parse.urlencode({
            "db": "pubmed", 
            "term": query, 
            "retmode": "json", 
            "retmax": max_results
        })
        
        req = urllib.request.Request(
            f"{base_url}esearch.fcgi?{search_params}",
            headers={'User-Agent': 'MedicalLiteratureAssistant/1.0 (contact: test@example.com)'}
        )
        
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            
        pmids = data.get("esearchresult", {}).get("idlist", [])
        print(f"Found PMIDs: {pmids}")
        
        if not pmids:
            return []

        # 2. Fetch details for found PMIDs
        fetch_params = urllib.parse.urlencode({
            "db": "pubmed", 
            "id": ",".join(pmids), 
            "retmode": "xml"
        })
        
        req_fetch = urllib.request.Request(
            f"{base_url}efetch.fcgi?{fetch_params}",
            headers={'User-Agent': 'MedicalLiteratureAssistant/1.0 (contact: test@example.com)'}
        )
        
        with urllib.request.urlopen(req_fetch) as response:
            xml_data = response.read().decode()

        papers = []
        root = ET.fromstring(xml_data)
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID")
            title = article.findtext(".//ArticleTitle") or "Untitled"
            abstract = "".join([elem.text for elem in article.findall(".//AbstractText") if elem.text])
            papers.append({
                "pmid": pmid, 
                "title": title, 
                "abstract": abstract or "No abstract available."
            })
            
        return papers

    except Exception as e:
        print(f"Error fetching from PubMed: {e}")
        return []

if __name__ == "__main__":
    print("Testing PubMed Integration...")
    results = search_pubmed("seizure")
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))