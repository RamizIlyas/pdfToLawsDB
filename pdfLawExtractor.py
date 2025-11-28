import PyPDF2
import re
from pymongo import MongoClient
from datetime import datetime

class PakistanPenalCodeExtractor:
    def __init__(self, mongodb_uri="mongodb://localhost:27017/", db_name="pakistan_law_db"):
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.sections_collection = self.db["penal_code_sections"]
        self.chapters_collection = self.db["penal_code_chapters"]
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def parse_sections(self, text):
        """Parse the PDF text to extract sections and their content"""
        sections = []
        chapters = []
        
        # Split text into lines
        lines = text.split('\n')
        
        current_chapter = None
        current_section = None
        current_content = ""
        chapter_content = ""
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for chapter headers
            chapter_match = re.match(r'CHAPTER\s+([IVXLCDM]+)(?:\s+OF\s+(.*))?', line, re.IGNORECASE)
            if chapter_match:
                # Save previous chapter if exists
                if current_chapter:
                    chapters.append({
                        "chapter_number": current_chapter["number"],
                        "chapter_title": current_chapter["title"],
                        "content": chapter_content.strip(),
                        "sections": current_chapter["sections"]
                    })
                
                chapter_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2) if chapter_match.group(2) else self.get_chapter_title(chapter_num)
                
                current_chapter = {
                    "number": chapter_num,
                    "title": chapter_title,
                    "sections": []
                }
                chapter_content = ""
                i += 1
                continue
            
            # Check for section headers
            section_match = re.match(r'^(\d+[A-Z]*)\.\s+(.*)$', line)
            if section_match:
                # Save previous section if exists
                if current_section:
                    section_data = {
                        "section_number": current_section["number"],
                        "section_title": current_section["title"],
                        "content": current_content.strip(),
                        "chapter": current_chapter["number"] if current_chapter else "Unknown",
                        "page_reference": self.find_page_reference(text, current_section["number"])
                    }
                    sections.append(section_data)
                    
                    if current_chapter:
                        current_chapter["sections"].append(section_data["section_number"])
                
                section_num = section_match.group(1)
                section_title = section_match.group(2)
                
                current_section = {
                    "number": section_num,
                    "title": section_title
                }
                current_content = ""
                i += 1
                continue
            
            # Accumulate content
            if current_section:
                current_content += line + " "
            elif current_chapter:
                chapter_content += line + " "
            
            i += 1
        
        # Add the last section and chapter
        if current_section:
            section_data = {
                "section_number": current_section["number"],
                "section_title": current_section["title"],
                "content": current_content.strip(),
                "chapter": current_chapter["number"] if current_chapter else "Unknown",
                "page_reference": self.find_page_reference(text, current_section["number"])
            }
            sections.append(section_data)
            
            if current_chapter:
                current_chapter["sections"].append(section_data["section_number"])
        
        if current_chapter:
            chapters.append({
                "chapter_number": current_chapter["number"],
                "chapter_title": current_chapter["title"],
                "content": chapter_content.strip(),
                "sections": current_chapter["sections"]
            })
        
        return sections, chapters
    
    def get_chapter_title(self, chapter_num):
        """Get chapter title based on chapter number"""
        chapter_titles = {
            "I": "INTRODUCTION",
            "II": "GENERAL EXPLANATIONS",
            "III": "OF PUNISHMENTS",
            "IV": "GENERAL EXCEPTIONS",
            "V": "OF ABETMENT",
            "VA": "CRIMINAL CONSPIRACY",
            "VI": "OF OFFENCES AGAINST THE STATE",
            "VII": "OF OFFENCES RELATING TO THE ARMY, NAVY AND AIR FORCE",
            "VIII": "OF OFFENCES AGAINST THE PUBLIC TRANQUILLITY",
            "IX": "OF OFFENCES BY OR RELATING TO PUBLIC SERVANTS",
            "IXA": "OF OFFENCES RELATING TO ELECTIONS",
            "X": "OF CONTEMPTS OF THE LAWFUL AUTHORITY OF PUBLIC SERVANTS",
            "XI": "OF FALSE EVIDENCE AND OFFENCES AGAINST PUBLIC JUSTICE",
            "XII": "OF OFFENCES RELATING TO COIN AND GOVERNMENT STAMPS",
            "XIII": "OF OFFENCES RELATING TO WEIGHTS AND MEASURES",
            "XIV": "OF OFFENCES AFFECTING THE PUBLIC HEALTH, SAFETY, CONVENIENCE, DECENCY AND MORALS",
            "XV": "OF OFFENCES RELATING TO RELIGION",
            "XVI": "OF OFFENCES AFFECTING THE HUMAN BODY",
            "XVI-A": "OF WRONGFUL RESTRAINT & WRONGFUL CONFINEMENT"
        }
        return chapter_titles.get(chapter_num, f"CHAPTER {chapter_num}")
    
    def find_page_reference(self, text, section_number):
        """Find the page where a section appears"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if section_number + '.' in line:
                # Estimate page number (assuming ~50 lines per page)
                return i // 50 + 1
        return None
    
    def store_in_mongodb(self, sections, chapters):
        """Store extracted data in MongoDB"""
        # Clear existing collections
        self.sections_collection.delete_many({})
        self.chapters_collection.delete_many({})
        
        # Add metadata and timestamps
        for section in sections:
            section["created_at"] = datetime.utcnow()
            section["updated_at"] = datetime.utcnow()
            section["document_type"] = "penal_code_section"
        
        for chapter in chapters:
            chapter["created_at"] = datetime.utcnow()
            chapter["updated_at"] = datetime.utcnow()
            chapter["document_type"] = "penal_code_chapter"
        
        # Insert into MongoDB
        if sections:
            result_sections = self.sections_collection.insert_many(sections)
            print(f"Inserted {len(result_sections.inserted_ids)} sections")
        
        if chapters:
            result_chapters = self.chapters_collection.insert_many(chapters)
            print(f"Inserted {len(result_chapters.inserted_ids)} chapters")
    
    def create_indexes(self):
        """Create indexes for better query performance"""
        self.sections_collection.create_index([("section_number", 1)])
        self.sections_collection.create_index([("chapter", 1)])
        self.sections_collection.create_index([("section_title", "text"), ("content", "text")])
        
        self.chapters_collection.create_index([("chapter_number", 1)])
        print("Indexes created successfully")
    
    def process_pdf(self, pdf_path):
        """Main method to process PDF and store in MongoDB"""
        print("Extracting text from PDF...")
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print("No text extracted from PDF")
            return
        
        print("Parsing sections and chapters...")
        sections, chapters = self.parse_sections(text)
        
        print(f"Found {len(sections)} sections and {len(chapters)} chapters")
        
        print("Storing in MongoDB...")
        self.store_in_mongodb(sections, chapters)
        
        print("Creating indexes...")
        self.create_indexes()
        
        print("Process completed successfully!")
    
    def query_sections(self, section_number=None, chapter=None, search_term=None):
        """Query sections from MongoDB"""
        query = {}
        
        if section_number:
            query["section_number"] = section_number
        if chapter:
            query["chapter"] = chapter
        if search_term:
            query["$text"] = {"$search": search_term}
        
        return list(self.sections_collection.find(query))
    
    def close_connection(self):
        """Close MongoDB connection"""
        self.client.close()

# Usage example
if __name__ == "__main__":
    # Initialize the extractor
    extractor = PakistanPenalCodeExtractor()
    
    try:
        # Process the PDF file
        extractor.process_pdf(".\law_pdfs\Pakistan Penal Code.pdf")
        
        # Example queries
        print("\n--- Example Queries ---")
        
        # Query specific section
        section_302 = extractor.query_sections(section_number="302")
        if section_302:
            print(f"Section 302: {section_302[0]['section_title']}")
            print(f"Content preview: {section_302[0]['content'][:200]}...")
        
        # Query sections from specific chapter
        chapter_16_sections = extractor.query_sections(chapter="XVI")
        print(f"Found {len(chapter_16_sections)} sections in Chapter XVI")
        
        # Search for sections containing specific terms
        murder_sections = extractor.query_sections(search_term="murder")
        print(f"Found {len(murder_sections)} sections related to 'murder'")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        extractor.close_connection()