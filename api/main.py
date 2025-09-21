import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
import torch

from api.routes.v1_router import v1_router
from api.settings import api_settings
from api.state import ml_models  # <-- Import from the neutral state file
from .core.embedding_manager import OptimizedEmbeddingManager


async def process_startup_templates():
    """Process all ideal contract templates on startup - only new/modified files"""
    try:
        # Import here to avoid circular imports
        from api.core.ideal_contract_manager import IdealContractManager, ContractCategory
        from api.core.document_processor import process_document_stream
        
        templates_folder = os.path.join(os.getcwd(), "ideal_contract_templates")
        
        if not os.path.exists(templates_folder):
            print(f"⚠️ Templates folder not found: {templates_folder}")
            return
        
        # Get all PDF files from the folder (ignore README and other files)
        pdf_files = [f for f in os.listdir(templates_folder) 
                    if f.lower().endswith('.pdf') and not f.lower().startswith('readme')]
        
        if not pdf_files:
            print("📁 No PDF templates found in ideal_contract_templates folder")
            return
        
        # Get embedding manager
        embedding_manager = ml_models.get("embedding_manager")
        if not embedding_manager:
            print("❌ Embedding manager not available for template processing")
            return
        
        ideal_manager = IdealContractManager()
        
        # Get existing templates to check what's already processed
        existing_templates = ideal_manager.list_ideal_contracts()
        
        # Create a set of existing template sources for quick lookup
        # We'll use the filename as stored in the metadata
        existing_sources = set()
        for template in existing_templates:
            if 'source_file' in template:
                existing_sources.add(template['source_file'])
        
        # Filter files to only process new ones
        files_to_process = []
        for pdf_file in pdf_files:
            if pdf_file not in existing_sources:
                files_to_process.append(pdf_file)
            else:
                # Check if file was modified since last processing
                file_path = os.path.join(templates_folder, pdf_file)
                file_mtime = os.path.getmtime(file_path)
                
                # Find the template for this file
                template_for_file = None
                for template in existing_templates:
                    if template.get('source_file') == pdf_file:
                        template_for_file = template
                        break
                
                if template_for_file:
                    # Check if file was modified after template creation
                    template_created = template_for_file.get('created_at')
                    if template_created:
                        try:
                            from datetime import datetime
                            created_dt = datetime.fromisoformat(template_created.replace('Z', '+00:00'))
                            file_dt = datetime.fromtimestamp(file_mtime)
                            
                            if file_dt > created_dt:
                                print(f"� File {pdf_file} was modified since last processing")
                                files_to_process.append(pdf_file)
                            else:
                                print(f"✅ File {pdf_file} already processed and unchanged")
                        except:
                            # If we can't parse dates, process it to be safe
                            files_to_process.append(pdf_file)
                    else:
                        # No creation date, process it
                        files_to_process.append(pdf_file)
                else:
                    # Shouldn't happen, but process it to be safe
                    files_to_process.append(pdf_file)
        
        if not files_to_process:
            print("✅ All ideal contract templates are already up-to-date")
            return
        
        processed_count = 0
        print(f"📁 Processing {len(files_to_process)} new/modified ideal contract templates...")
        
        for pdf_file in files_to_process:
            try:
                file_path = os.path.join(templates_folder, pdf_file)
                print(f"📄 Processing: {pdf_file}")
                
                # Extract category from filename (before first underscore)
                filename_parts = pdf_file.replace('.pdf', '').split('_')
                if len(filename_parts) < 2:
                    print(f"⚠️ Skipping {pdf_file} - Invalid filename format. Use: category_description.pdf")
                    continue
                
                category_name = filename_parts[0].lower()
                description = "_".join(filename_parts[1:]).replace('_', ' ').title()
                
                # Validate category
                try:
                    category_enum = ContractCategory(category_name)
                except ValueError:
                    print(f"⚠️ Skipping {pdf_file} - Unknown category: {category_name}")
                    continue
                
                # If this file was already processed, delete the old template first
                if pdf_file in existing_sources:
                    print(f"🔄 Updating existing template for {pdf_file}")
                    # Find and delete the old template
                    for template in existing_templates:
                        if template.get('source_file') == pdf_file:
                            old_id = template.get('id')
                            if old_id:
                                ideal_manager.delete_ideal_contract(old_id)
                                print(f"🗑️ Deleted old template: {old_id}")
                            break
                
                # Read and process the PDF
                document_url = f"template_folder://{pdf_file}"
                
                # Create async iterator from file content
                async def file_iterator():
                    with open(file_path, 'rb') as f:
                        chunk_size = 8192
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            yield chunk
                
                chunks = await process_document_stream(
                    document_url,
                    file_iterator()
                )
                
                if not chunks:
                    print(f"⚠️ Skipping {pdf_file} - Could not extract text")
                    continue
                
                # Combine all chunks to get full document text
                full_text = "\n".join([chunk[0] for chunk in chunks])
                
                if len(full_text.strip()) < 100:
                    print(f"⚠️ Skipping {pdf_file} - Text too short ({len(full_text)} chars)")
                    continue
                
                # Create embedding for the full document
                embedding = embedding_manager.encode_batch([full_text])[0]
                
                # Create template structure
                title = f"{category_name.title()} Template - {description}"
                template_data = {
                    "category": category_enum,
                    "title": title,
                    "description": f"Legal template for {category_name} contracts - {description}",
                    "essential_clauses": [
                        {
                            "name": "standard_clauses",
                            "description": f"Standard clauses found in {category_name} contracts",
                            "importance": 10,
                            "keywords": [category_name, "contract", "agreement", "terms"],
                            "required": True
                        }
                    ],
                    "risk_factors": [
                        {
                            "name": "missing_protections",
                            "description": f"Important {category_name} protections missing from user contract",
                            "risk_level": "high",
                            "keywords": ["missing", "protection", "rights"],
                            "penalty_score": -25
                        }
                    ],
                    "compliance_requirements": [
                        {
                            "name": "legal_standards",
                            "description": f"Must meet legal standards for {category_name} contracts",
                            "required": True,
                            "keywords": ["legal", "compliance", "standard"]
                        }
                    ],
                    "scoring_weights": {
                        "essential_clauses": 0.6,
                        "risk_factors": 0.3,
                        "compliance": 0.1
                    }
                }
                
                # Store the ideal contract
                template_id = ideal_manager.store_ideal_contract(
                    category=template_data["category"],
                    title=template_data["title"],
                    description=template_data["description"],
                    essential_clauses=template_data["essential_clauses"],
                    risk_factors=template_data["risk_factors"],
                    compliance_requirements=template_data["compliance_requirements"],
                    scoring_weights=template_data["scoring_weights"],
                    embedding=embedding,
                    created_by="startup_processor",
                    source_file=pdf_file
                )
                
                processed_count += 1
                print(f"✅ Processed: {pdf_file} -> {template_id}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                continue
        
        if processed_count > 0:
            print(f"✅ Successfully processed {processed_count} new/modified ideal contract templates")
        else:
            print("⚠️ No new templates were successfully processed")
            
    except Exception as e:
        print(f"❌ Error during startup template processing: {e}")

# A simple dictionary to hold the loaded model manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- Server starting up ---")

    # Instantiate the optimized embedding manager
    manager = OptimizedEmbeddingManager()
    print(f"Embedding manager initialized on device: {manager.device}")

    # Store manager in shared state
    ml_models["embedding_manager"] = manager
    print("✅ Embedding manager loaded successfully.")
    
    # Initialize AI model for contract recommendations
    try:
        from api.core.simple_ai_model import initialize_ai_model
        ai_model = initialize_ai_model()
        print("✅ AI recommendation model loaded successfully.")
    except Exception as e:
        print(f"⚠️ AI model initialization failed: {str(e)}")
        print("Guardian Score will use static recommendations.")
    
    # Process ideal contract templates on startup
    print("🔄 Processing ideal contract templates...")
    await process_startup_templates()
    
    print(ml_models)

    yield

    print("--- Server shutting down ---")
    ml_models.clear()


def create_app() -> FastAPI:
    """Create a FastAPI App"""
    app: FastAPI = FastAPI(
        title=api_settings.title,
        version=api_settings.version,
        docs_url="/docs" if api_settings.docs_enabled else None,
        redoc_url="/redoc" if api_settings.docs_enabled else None,
        openapi_url="/openapi.json" if api_settings.docs_enabled else None,
        lifespan=lifespan,  # <-- Use the new lifespan manager
    )

    app.include_router(v1_router, prefix="/api")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Allow frontend
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Check if frontend build exists and serve static files
    frontend_build_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "out")
    if os.path.exists(frontend_build_path):
        # Mount the static files
        app.mount("/static", StaticFiles(directory=os.path.join(frontend_build_path, "_next", "static")), name="static")
        
        # Serve the main app
        @app.get("/")
        async def serve_frontend():
            return FileResponse(os.path.join(frontend_build_path, "index.html"))
        
        # Handle all other routes by serving the frontend
        @app.get("/{path:path}")
        async def serve_frontend_routes(path: str):
            file_path = os.path.join(frontend_build_path, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            # For client-side routing, serve index.html
            return FileResponse(os.path.join(frontend_build_path, "index.html"))
    
    return app


app = create_app()
