from pathlib import Path

from ict_agent.skills import build_skill_prompt, load_skills, select_skills


def test_load_skills_from_project():
    skills_root = Path(__file__).resolve().parents[2] / "skills"
    skills = load_skills(skills_root)
    assert "core" in skills
    assert "cuda" in skills


def test_select_skills_for_shell_and_filesystem_intent():
    skills_root = Path(__file__).resolve().parents[2] / "skills"
    skills = load_skills(skills_root)
    selected = select_skills("read model.py and run compile command", skills)
    selected_names = {skill.name for skill in selected}
    assert "filesystem" in selected_names
    assert "shell" in selected_names
    assert "core" in selected_names
    assert build_skill_prompt(selected)
