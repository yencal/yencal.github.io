source "https://rubygems.org"

# Use GitHub Pages gem which includes Jekyll and common plugins
gem "github-pages", group: :jekyll_plugins

# Plugins for local development or version pinning
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.12"   # Generates RSS feed
  gem "jekyll-sitemap"           # Generates sitemap.xml
  gem "jekyll-seo-tag"           # Adds SEO meta tags
end

# Code highlighting
gem "rouge"

# Windows and JRuby support
platforms :windows, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance booster for watching directories on Windows
gem "wdm", "~> 0.1", platforms: [:windows]

# Lock `http_parser.rb` gem for JRuby
gem "http_parser.rb", "~> 0.6.0", platforms: [:jruby]

# Optional: pin your Jekyll version (uncomment if you want)
# gem "jekyll", "~> 4.4.1"

# Optional: pin your theme version (uncomment if you want)
# gem "minima", "~> 2.5"
