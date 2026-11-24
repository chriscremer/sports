# Updating pickleball results

The [live standings and schedule](https://chriscremer.ca/sports/pickleball/pickleball_standings.html#schedule) are maintained in [pickleball_standings.html](pickleball_standings.html). The page is static HTML. The video-analysis tools in this directory are unrelated to standings updates.

## Automatic refresh

The standings generator lives in the separate SVA repository. Fetch the latest results and rebuild the local page with:

```bash
python3 /Users/chriscremer/code/SVA/SVA/Z_other_code/pickleball/build_standings.py
```

Review the result locally. The steps below describe how to verify or update the page if the generator is unavailable.

## Manual verification

1. Open the [Off Limit Sports LeagueApps schedule](https://offlimitsports.leagueapps.com/leagues/5080557/schedule) for Fall 2026 Monday tiered pairs. LeagueApps may require a member sign-in. Use an authorized session, or ask for a saved copy of the schedule or a score export if access is unavailable. For a new season, confirm the league ID and update the source link on the page.
2. Compare the official schedule with the **All Results** table. Record each completed game separately with its LeagueApps game ID, Toronto date and start time, both teams and scores in source order, and court. Match existing rows by game ID so score corrections replace the original entry. Leave games with no posted score pending. A posted 0–0 is distinct from a blank score.
3. Update **All Results**, the individual team cards, standings, Elo table and history plot, and upcoming sections in the same change. Keep the results newest first. In standings, count each scored game, including forfeits, and calculate win percentage as (wins + half of ties) / games played. Rank by win percentage, then wins, then point differential. The page lists teams with no upcoming games separately.
4. For Elo, replay the completed games in date and game-ID order from 1500 with K = 32. Exclude 5–0 and 0–5 forfeits from both Elo and the Elo record. Use actual outcomes of 1, 0.5, or 0 for a win, tie, or loss and the standard Elo expectation, `1 / (1 + 10 ** ((opponentRating - rating) / 400))`. Use pre-game ratings when updating both teams. Refresh the latest week's changes, match ratings, upcoming win probabilities, and team rating colours. Keep the existing tiers unless the league has issued new assignments.
5. Reconcile the upcoming fixtures with LeagueApps, including dates, times, and courts. The **Schedule** tab shows the featured teams and **Next Week** shows the next league night. Check the completed game count, unique IDs, team records, and links, then update the page timestamp in Toronto time and preview the tabs. After publishing to `main`, check the live page for the new results and schedule.

For an example, see the [November 16 results update](https://github.com/chriscremer/sports/commit/4b736588a9593f910297ed07b9838e18bfbb5683).

## Data sources and refresh details

The script fetches the [LeagueApps schedule JSON](https://offlimitsports.leagueapps.com/v1/sites/2415/programs/5080557/schedule) using the public API key stored in the script. The response contains team IDs, game times, scores, and game status. The script calculates standings and Elo from those results; 5–0 and 0–5 forfeits count in standings but do not change Elo.

Tier assignments are downloaded from the [league tier PDF](https://drive.google.com/file/d/1nNzovCzuVaU_5bl7Gs7XIRDJ-n9m14I5/view). PDF extraction requires `pdfplumber` or the Codex bundled Python. If tier extraction fails, the script prints a warning and uses its last saved tier data. The [playoff format PDF](https://drive.google.com/file/d/1jTT2xDvvhzGMiTaZqfH8fBKDum2CeRtD/view) is summarized in the generator's HTML template and should be checked manually for changes.

The HTML is a snapshot, so run the command again when new results are posted. The generator and template are in `/Users/chriscremer/code/SVA/SVA/Z_other_code/pickleball`, outside this repository. Review the generated page locally; the public site updates after the file is committed and pushed from the `sports` repository.
