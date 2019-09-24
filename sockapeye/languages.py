class LanguageResolver:
    def __init__(self):
        pass

    def available_languages(self):
        return {
            "languages": [
                {
                    "key":'en',
                    "name": 'English'
                },
                {
                    "key":'de',
                    "name": 'German'
                },
                {
                    "key":'fr',
                    "name": 'French'
                }
            ],
            "languagePairs": [
                {
                    "source": "de",
                    "targets": ["en", "fr"]
                },
                {
                    "source": "en",
                    "targets": ["de"]
                }
            ],
        }