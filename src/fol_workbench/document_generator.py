"""
Beautiful Document Generator

Generates beautiful, formatted documents from data using templates.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json


class DocumentGenerator:
    """Generates beautiful formatted documents."""
    
    def __init__(self):
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load document templates."""
        # HTML template
        self.templates['html'] = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .document {{
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metadata {{
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        .content {{
            margin-top: 20px;
        }}
        .formula {{
            background: #ecf0f1;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="document">
        <h1>{title}</h1>
        <div class="metadata">
            Generated: {date}
        </div>
        <div class="content">
            {content}
        </div>
    </div>
</body>
</html>"""
        
        # Markdown template
        self.templates['markdown'] = """# {title}

*Generated: {date}*

{content}
"""
    
    def generate_html(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate HTML document."""
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.templates['html'].format(
            title=title,
            date=date,
            content=content
        )
    
    def generate_markdown(self, title: str, content: str) -> str:
        """Generate Markdown document."""
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.templates['markdown'].format(
            title=title,
            date=date,
            content=content
        )
    
    def generate_thesis_document(self, thesis_data: Dict[str, Any]) -> str:
        """Generate beautiful thesis document."""
        content = f"""
        <h2>{thesis_data.get('title', 'Thesis')}</h2>
        <div class="formula">
            <strong>Statement:</strong><br>
            {thesis_data.get('statement', '')}
        </div>
        """
        
        if thesis_data.get('proof'):
            content += f"""
            <h3>Proof</h3>
            <div class="content">
                {thesis_data.get('proof', '')}
            </div>
            """
        
        if thesis_data.get('axioms_used'):
            content += """
            <h3>Axioms Used</h3>
            <ul>
            """
            for axiom_id in thesis_data.get('axioms_used', []):
                content += f"<li>{axiom_id}</li>"
            content += "</ul>"
        
        return self.generate_html(
            title=thesis_data.get('title', 'Thesis'),
            content=content
        )
    
    def generate_task_report(self, tasks: List[Dict[str, Any]]) -> str:
        """Generate task report document."""
        content = f"""
        <h2>Task Report</h2>
        <p>Total Tasks: {len(tasks)}</p>
        <table>
            <thead>
                <tr>
                    <th>Title</th>
                    <th>Status</th>
                    <th>Priority</th>
                    <th>Due Date</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for task in tasks:
            content += f"""
                <tr>
                    <td>{task.get('title', '')}</td>
                    <td>{task.get('status', '')}</td>
                    <td>{task.get('priority', '')}</td>
                    <td>{task.get('due_date', 'N/A')}</td>
                </tr>
            """
        
        content += """
            </tbody>
        </table>
        """
        
        return self.generate_html(title="Task Report", content=content)
    
    def generate_axiom_catalog(self, axioms: List[Dict[str, Any]]) -> str:
        """Generate axiom catalog document."""
        content = f"""
        <h2>Axiom Catalog</h2>
        <p>Total Axioms: {len(axioms)}</p>
        """
        
        for axiom in axioms:
            content += f"""
            <div class="formula">
                <h3>{axiom.get('name', 'Unnamed Axiom')}</h3>
                <p><strong>Formula:</strong> {axiom.get('formula', '')}</p>
                <p>{axiom.get('description', '')}</p>
            </div>
            """
        
        return self.generate_html(title="Axiom Catalog", content=content)
    
    def save_document(self, document: str, filepath: str, format: str = "html"):
        """Save document to file."""
        path = Path(filepath)
        if format == "html" and not path.suffix:
            path = path.with_suffix('.html')
        elif format == "markdown" and not path.suffix:
            path = path.with_suffix('.md')
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(document)
        
        return path
