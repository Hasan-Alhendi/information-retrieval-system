"""Human-readable semantic category definitions for guided retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryProfile:
    """A named semantic category represented by a natural-language description."""

    name: str
    label: str
    description: str


QUORA_CATEGORIES: tuple[CategoryProfile, ...] = (
    CategoryProfile(
        "technology",
        "Technology & Programming",
        "Questions about computers, programming, software, mobile applications, artificial intelligence, cybersecurity, and the internet.",
    ),
    CategoryProfile(
        "education",
        "Education & Learning",
        "Questions about studying, learning methods, schools, universities, courses, exams, teaching, and academic skills.",
    ),
    CategoryProfile(
        "health",
        "Health & Fitness",
        "Questions about medicine, diseases, symptoms, treatment, nutrition, exercise, fitness, and mental health.",
    ),
    CategoryProfile(
        "careers",
        "Careers & Work",
        "Questions about jobs, interviews, salaries, professions, workplace skills, business, and career development.",
    ),
    CategoryProfile(
        "finance",
        "Finance & Money",
        "Questions about money, banking, investment, saving, debt, markets, personal finance, and economic decisions.",
    ),
    CategoryProfile(
        "relationships",
        "Relationships & Family",
        "Questions about love, marriage, friendship, family, parenting, communication, and social relationships.",
    ),
    CategoryProfile(
        "travel",
        "Travel & Places",
        "Questions about countries, cities, tourism, immigration, transportation, flights, hotels, and travel planning.",
    ),
    CategoryProfile(
        "science",
        "Science & Mathematics",
        "Questions about physics, chemistry, biology, mathematics, astronomy, engineering, and scientific explanations.",
    ),
    CategoryProfile(
        "politics",
        "Politics & Society",
        "Questions about governments, elections, political systems, public policy, history, culture, and social issues.",
    ),
    CategoryProfile(
        "religion_philosophy",
        "Religion & Philosophy",
        "Questions about religion, spirituality, ethics, beliefs, philosophy, meaning, and moral reasoning.",
    ),
    CategoryProfile(
        "entertainment",
        "Entertainment & Media",
        "Questions about movies, television, books, music, games, celebrities, sports, and popular culture.",
    ),
    CategoryProfile(
        "daily_life",
        "Daily Life & Practical Advice",
        "Questions about everyday problems, habits, food, shopping, home, personal improvement, and practical advice.",
    ),
)


TOUCHE_CATEGORIES: tuple[CategoryProfile, ...] = (
    CategoryProfile(
        "politics",
        "Politics & Government",
        "Arguments about elections, governments, democracy, political systems, public administration, and political power.",
    ),
    CategoryProfile(
        "environment",
        "Environment & Energy",
        "Arguments about climate change, pollution, conservation, energy sources, sustainability, and environmental policy.",
    ),
    CategoryProfile(
        "technology",
        "Technology & Digital Society",
        "Arguments about artificial intelligence, automation, the internet, privacy, social media, and digital technology.",
    ),
    CategoryProfile(
        "education",
        "Education",
        "Arguments about schools, universities, teaching, curriculum, student assessment, access, and education policy.",
    ),
    CategoryProfile(
        "health",
        "Health & Medicine",
        "Arguments about healthcare, medicine, public health, treatment, vaccination, nutrition, and medical ethics.",
    ),
    CategoryProfile(
        "economy",
        "Economy & Work",
        "Arguments about taxes, trade, employment, markets, welfare, inequality, business, and economic policy.",
    ),
    CategoryProfile(
        "law",
        "Law & Rights",
        "Arguments about laws, courts, crime, punishment, civil rights, regulation, justice, and legal responsibility.",
    ),
    CategoryProfile(
        "ethics",
        "Ethics & Human Values",
        "Arguments about morality, fairness, responsibility, freedom, human dignity, and ethical dilemmas.",
    ),
    CategoryProfile(
        "social_issues",
        "Social Issues",
        "Arguments about family, gender, discrimination, migration, culture, social equality, and community life.",
    ),
    CategoryProfile(
        "science",
        "Science & Research",
        "Arguments about scientific evidence, research, engineering, space, biology, and the social use of science.",
    ),
    CategoryProfile(
        "religion",
        "Religion & Belief",
        "Arguments about religion, faith, secularism, religious freedom, and the public role of belief.",
    ),
    CategoryProfile(
        "public_policy",
        "Public Policy",
        "Arguments about government programs, regulation, public services, national priorities, and policy trade-offs.",
    ),
)


CATEGORY_PROFILES: dict[str, tuple[CategoryProfile, ...]] = {
    "quora": QUORA_CATEGORIES,
    "touche2020-v2": TOUCHE_CATEGORIES,
}


def get_category_profiles(dataset_name: str) -> tuple[CategoryProfile, ...]:
    """Return the guided semantic categories configured for a dataset."""
    try:
        return CATEGORY_PROFILES[dataset_name]
    except KeyError as exc:
        supported = ", ".join(sorted(CATEGORY_PROFILES))
        raise ValueError(
            f"Guided categories are unavailable for '{dataset_name}'. Supported: {supported}"
        ) from exc
