import os
from langchain_huggingface import HuggingFaceEmbeddings
from agents.sql_agent import SQLAgentTools
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

golden_set = {
    "Properties and Plots": [
        "What crops exist and what are their areas across all properties?",
        "What is the vineyard area in the municipality of Vila Real?",
        "What is the area of Touriga Nacional plots?",
        "What is the vineyard area under Integrated Production?",
        "List plots by training system and area",
        "What rootstocks are used in property Quinta do Vale?",
        "List irrigation types by property, plot, and area",
        "List plots with training system, trellising type, and planting year",
        "Calculate estimated plant count for plot A1 in property Quinta Norte",
        "Show average altitude and slope per property",
    ],
    "Cultural Practices": [
        "List phytosanitary applications during 2025",
        "What was the pruning period, cost, and time for property Quinta Sul?",
        "Where was AGRO-STAR applied during 2025?",
        "Where was fertilizer 7-14-14 applied and in what quantities during 2025?",
        "List days worked by Contractor 1 during 2025 grouped by month",
        "Show the history of all practices by Worker Maria Silva",
        "What cultural practices were done in 2025 with Crawler Tractor?",
        "Show hours and costs by practice type for 2025",
        "Show hours and costs by resource type for 2025",
        "Show production costs and total hours worked for 2025 grouped by month",
    ],
    "Cross-Reference": [
        "List cultural practices with costs and hours for property Quinta do Vale",
        "List cultural practices with costs and hours for plot A1",
        "List pruning practices with costs and hours grouped by variety for property Quinta do Vale",
        "Compare costs and hours by training system for vineyards",
        "Show costs and hours by resource type for each property",
        "Create a calendar with hours worked by HR in cultural practices",
        "Create a calendar with equipment usage hours",
    ],
    "Cellar": [
        "List all red wine fermentations for 2025",
        "List all white wine fermentations grouped by variety for 2025",
        "List all operations for lot DOC Red",
        "Show costs and hours for lot DOC Red",
        "List wine quantities by wine type",
        "List wine quantities by production phase",
        "List wine losses by cellar phase",
        "List usage history of consumable Bottle 0.75",
        "Create a calendar with hours worked in cellar operations for HR",
        "Calculate costs and hours for all lots registered in 2025",
    ]
}

def main():
    print("Loading embeddings and SQL agent...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        encode_kwargs={"normalize_embeddings": True},
    )

    agent = SQLAgentTools(embeddings)
    agent.add_database("Operations", os.getenv("DB_OPERATIONS_URL"))
    agent.add_database("Plots", os.getenv("DB_PLOTS_URL"))
    agent.add_database("Cellar", os.getenv("DB_CELLAR_URL"))
    agent.setup()

    print("\n" + "="*80)
    print("SCHEMA RAG TEST - GOLDEN SET")
    print("="*80)

    with open("schema_rag_test_result.txt", "w", encoding="utf-8") as f:
        for category, questions in golden_set.items():
            print(f"\n--- {category} ---")
            f.write(f"\n{'='*50}\nCATEGORY: {category}\n{'='*50}\n")

            for i, question in enumerate(questions, 1):
                schema_text, tables_found = agent._prefetch_schemas(question)
                tables_str = ", ".join([f"{db}.{t}" for db, t in tables_found])

                print(f"[{i:02d}] {question[:60]}... -> {len(tables_found)} tables")

                f.write(f"\nQ: {question}\n")
                if not tables_found:
                    f.write("RAG: FAILED - No tables found!\n")
                else:
                    f.write(f"Retrieved tables ({len(tables_found)}):\n")
                    for db, t in tables_found:
                        f.write(f"  - [{db}] {t}\n")
                f.write("-" * 40 + "\n")

    print("\nTest complete! Details saved to 'schema_rag_test_result.txt'")

if __name__ == "__main__":
    main()
