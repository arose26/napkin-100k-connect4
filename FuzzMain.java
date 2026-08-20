/* Venue-parity fuzz driver (verification artifact, not the deliverable).
 *
 * Runs the OFFICIAL CodinGame Connect 4 referee (github.com/AshKcg/cg-multi-connect4,
 * built locally with its own SDK) headlessly, with napkin_c4.py's CG protocol adapter
 * playing BOTH seats on a random policy. The adapter rebuilds the position from the
 * referee's own board rows every turn and recomputes the valid-action set from its own
 * rules, printing PARITY MISMATCH to stderr on any divergence from the set the referee
 * sent -- including whether the STEAL is offered, which is the rule this arena adds.
 * This driver surfaces those lines plus the referee's real scores, per game.
 *
 * Run from the referee repo (see README "Repro"):
 *   javac -cp target/classes:<runner deps> -d target/fuzz FuzzMain.java
 *   java  -cp target/fuzz:target/classes:<runner deps> FuzzMain <games> <seed0>
 * Output: one "GAME <seed> scores={0=s0, 1=s1} predict=<agent>:PREDICT ..." line per
 * game (+ any PARITY lines), then "TOTAL parity_mismatch_lines=N". The README's Repro
 * block asserts N==0.
 */
import com.codingame.gameengine.runner.MultiplayerGameRunner;
import com.codingame.gameengine.runner.simulate.GameResult;

import java.util.List;
import java.util.Map;

public class FuzzMain {
    public static void main(String[] args) throws Exception {
        int games = Integer.parseInt(args[0]);
        long seed0 = Long.parseLong(args[1]);
        String py = System.getenv().getOrDefault("NAPKIN_C4_PY",
                System.getProperty("user.dir") + "/napkin_c4.py");
        String python = System.getenv().getOrDefault("NAPKIN_PYTHON", "python3");
        // optional custom agent commands (args 2/3); "SEEDn" is replaced per game
        String agent0 = args.length > 2 ? args[2]
                : python + " " + py + " cg --policy random --seed SEED0";
        String agent1 = args.length > 3 ? args[3]
                : python + " " + py + " cg --policy random --seed SEED1";
        int parityLines = 0;

        for (int g = 0; g < games; g++) {
            long seed = seed0 + g;
            MultiplayerGameRunner runner = new MultiplayerGameRunner();
            runner.setSeed(seed);
            runner.addAgent(agent0.replace("SEED0", Long.toString(seed * 2)));
            runner.addAgent(agent1.replace("SEED1", Long.toString(seed * 2 + 1)));
            GameResult res = runner.simulate();

            String predict = null;
            for (Map.Entry<String, List<String>> e : res.errors.entrySet()) {
                for (String chunk : e.getValue()) {
                    if (chunk == null) continue;
                    for (String line : chunk.split("\n")) {
                        if (line.startsWith("PREDICT")) {
                            predict = e.getKey() + ":" + line;
                        } else if (line.contains("PARITY")) {
                            parityLines++;
                            System.out.println("GAME " + seed + " " + line);
                        }
                    }
                }
            }
            System.out.println("GAME " + seed + " scores=" + res.scores
                    + " predict=" + predict);
        }
        System.out.println("TOTAL parity_mismatch_lines=" + parityLines);
    }
}
