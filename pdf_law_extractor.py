# pdf_law_extractor.py
import fitz  # PyMuPDF
import re
import os
from pymongo import MongoClient
from datetime import datetime
import spacy
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class LawPDFExtractor:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="legal_assistant"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.laws_collection = self.db.laws
        self.nlp = spacy.load("en_core_web_sm")
        
        # Pakistani law-specific patterns
        self.section_patterns = [
            r'Section\s+(\d+[A-Z]*)\s*(?:of\s*)?(?:PPC|Pakistan Penal Code|PPC\s*\d+)',
            r'Section\s+(\d+[A-Z]*)\s*-\s*([^\.]+)',
            r'(\d+[A-Z]*)\s*PPC',
            r'Section\s+(\d+[A-Z]*)',
            r'§\s*(\d+[A-Z]*)'
        ]
        
        self.act_patterns = [
            r'Pakistan Penal Code',
            r'PPC',
            r'Code of Criminal Procedure',
            r'CrPC',
            r'Code of Civil Procedure', 
            r'CPC',
            r'Constitution of Pakistan',
            r'Qanun-e-Shahadat'
        ]

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text):
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'Page\s*\d+\s*of\s*\d+', '', text)
        
        return text.strip()

    def identify_act_type(self, text):
        """Identify the type of legal act from text"""
        text_upper = text.upper()
        
        act_mapping = {
            'PAKISTAN PENAL CODE': 'PPC',
            'PPC': 'PPC', 
            'CODE OF CRIMINAL PROCEDURE': 'CrPC',
            'CRPC': 'CrPC',
            'CODE OF CIVIL PROCEDURE': 'CPC',
            'CPC': 'CPC',
            'CONSTITUTION OF PAKISTAN': 'Constitution',
            'QANUN-E-SHAHADAT': 'Qanun-e-Shahadat'
        }
        
        for pattern, act_code in act_mapping.items():
            if pattern in text_upper:
                return act_code
        return "Unknown"

    def extract_sections(self, text):
        """Extract law sections from text"""
        sections = []
        text = self.clean_text(text)
        
        # Split by potential section boundaries
        section_delimiters = [
            r'Section\s+\d+[A-Z]*',
            r'\d+[A-Z]*\.\s+',
            r'ARTICLE\s+\d+',
            r'§\s*\d+[A-Z]*'
        ]
        
        # Use multiple patterns to find sections
        for pattern in self.section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                section_start = match.start()
                # Find the end of this section (start of next section or end of text)
                next_section = re.search(r'Section\s+\d+[A-Z]*', text[section_start + 10:])
                if next_section:
                    section_end = section_start + 10 + next_section.start()
                else:
                    section_end = len(text)
                
                section_text = text[section_start:section_end].strip()
                
                # Extract section number
                section_num_match = re.search(r'(\d+[A-Z]*)', match.group(0))
                section_num = section_num_match.group(1) if section_num_match else "Unknown"
                
                # Extract title (first sentence after section number)
                title_match = re.search(r'Section\s+\d+[A-Z]*\s*[–\-]\s*([^\.]+\.?)', section_text)
                title = title_match.group(1).strip() if title_match else f"Section {section_num}"
                
                sections.append({
                    'section_number': section_num,
                    'title': title,
                    'content': section_text,
                    'act_type': self.identify_act_type(section_text)
                })
        
        return sections

    def extract_penalty_info(self, text):
        """Extract penalty information from section text"""
        penalty_keywords = [
            'punishable', 'punishment', 'imprisonment', 'fine', 
            'penalty', 'sentenced', 'liable to'
        ]
        
        penalties = []
        text_lower = text.lower()
        
        # Look for penalty patterns
        penalty_patterns = [
            r'punishable with imprisonment(?:[^.]{0,200}?)',
            r'imprisonment(?:.*?)for(?:.*?)\d+[^.]{0,100}',
            r'fine(?:.*?)which may extend to[^.]{0,100}',
            r'liable to(?:[^.]{0,200}?)'
        ]
        
        for pattern in penalty_patterns:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                penalty_text = match.group(0).strip()
                if len(penalty_text) > 10:  # Filter out too short matches
                    penalties.append(penalty_text)
        
        return penalties

    def extract_procedure_info(self, text):
        """Extract procedural information"""
        procedure_keywords = [
            'procedure', 'filed', 'complaint', 'fir', 'court',
            'application', 'petition', 'appeal', 'suit'
        ]
        
        procedures = []
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in procedure_keywords):
                procedures.append(sentence.strip())
        
        return procedures

    def categorize_law(self, title, content):
        """Categorize law based on content"""
        content_lower = content.lower()
        
        categories = {
            'property': ['property', 'land', 'possession', 'trespass', 'ownership'],
            'criminal': ['murder', 'theft', 'assault', 'criminal', 'offence', 'penalty'],
            'family': ['marriage', 'divorce', 'inheritance', 'family', 'maintenance'],
            'civil': ['contract', 'agreement', 'civil', 'suit', 'compensation'],
            'constitutional': ['right', 'fundamental', 'constitution', 'article'],
            'commercial': ['business', 'trade', 'commercial', 'company']
        }
        
        matched_categories = []
        for category, keywords in categories.items():
            if any(keyword in content_lower for keyword in keywords):
                matched_categories.append(category)
        
        return matched_categories if matched_categories else ['general']

    def process_law_section(self, section_data):
        """Process individual law section and prepare for database"""
        penalties = self.extract_penalty_info(section_data['content'])
        procedures = self.extract_procedure_info(section_data['content'])
        categories = self.categorize_law(section_data['title'], section_data['content'])
        
        law_document = {
            'section_number': section_data['section_number'],
            'title': section_data['title'],
            'act_type': section_data['act_type'],
            'content': section_data['content'],
            'penalties': penalties,
            'procedures': procedures,
            'categories': categories,
            'tags': self.generate_tags(section_data['title'], section_data['content']),
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'source': 'pdf_extraction'
        }
        
        return law_document

    def generate_tags(self, title, content):
        """Generate search tags from title and content"""
        doc = self.nlp(content[:500])  # Process first 500 characters for efficiency
        
        # Extract nouns and proper nouns as tags
        tags = set()
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 3:
                tags.add(token.text.lower())
        
        # Add words from title
        title_words = re.findall(r'\b[A-Za-z]{4,}\b', title)
        tags.update([word.lower() for word in title_words])
        
        return list(tags)[:10]  # Limit to 10 tags

    def save_to_mongodb(self, law_documents):
        """Save law documents to MongoDB"""
        try:
            # Remove duplicates based on section_number and act_type
            unique_documents = []
            seen = set()
            
            for doc in law_documents:
                key = (doc['section_number'], doc['act_type'])
                if key not in seen:
                    seen.add(key)
                    unique_documents.append(doc)
            
            # Insert into MongoDB
            if unique_documents:
                result = self.laws_collection.insert_many(unique_documents)
                logger.info(f"Successfully inserted {len(result.inserted_ids)} law sections into MongoDB")
                return len(result.inserted_ids)
            else:
                logger.warning("No unique documents to insert")
                return 0
                
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {str(e)}")
            return 0

    def process_pdf_directory(self, pdf_directory):
        """Process all PDF files in a directory"""
        all_law_documents = []
        
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                logger.info(f"Processing PDF: {filename}")
                
                # Extract text from PDF
                text = self.extract_text_from_pdf(pdf_path)
                if not text:
                    continue
                
                # Extract sections
                sections = self.extract_sections(text)
                logger.info(f"Found {len(sections)} sections in {filename}")
                
                # Process each section
                for section in sections:
                    law_doc = self.process_law_section(section)
                    all_law_documents.append(law_doc)
        
        # Save to MongoDB
        saved_count = self.save_to_mongodb(all_law_documents)
        return saved_count

    def validate_extraction(self):
        """Validate the extraction by checking database contents"""
        total_laws = self.laws_collection.count_documents({})
        acts_count = self.laws_collection.aggregate([
            {"$group": {"_id": "$act_type", "count": {"$sum": 1}}}
        ])
        
        logger.info(f"Total laws in database: {total_laws}")
        for act in acts_count:
            logger.info(f"  {act['_id']}: {act['count']} sections")

def main():
    """Main function to run the extraction"""
    
    # Initialize extractor
    extractor = LawPDFExtractor(
        mongo_uri=os.getenv("MONGO_URI", "mongodb://localhost:27017")
    )
    
    # Process PDF directory
    pdf_directory = "law_pdfs"  # Change this to your PDF directory path
    
    if not os.path.exists(pdf_directory):
        logger.error(f"PDF directory '{pdf_directory}' does not exist")
        return
    
    logger.info("Starting PDF law extraction...")
    
    # Process all PDFs in directory
    saved_count = extractor.process_pdf_directory(pdf_directory)
    
    logger.info(f"Extraction completed. Saved {saved_count} law sections to MongoDB.")
    
    # Validate extraction
    extractor.validate_extraction()

if __name__ == "__main__":
    main()