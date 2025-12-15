# Transformers and torch updates

Updating transformers library to `4.57.1` and torch to `2.9.0`. tt-xla and tt-forge-models changed needed are in branch [jazpur/transformers-uplift](https://github.com/tenstorrent/tt-xla/compare/main...jazpur/transformers-uplift). We should rebase this branch on top of main as frequently as possible to have it ready once perf issues are resolved.

The main tasks are to:
1. Keep the tt-xla changes up to date with `main`
2. Test the branch on the nightly tests
3. Address any new failure introduced by the rebase + uplifts

The following are just suggestions based on how I've been doing things. Feel free to do it your own way, as long as the three tasks I just mentioned are done.

## Rebasing

The branch has two commits, one for all the tt-xla changes and another for the tt-forge-models cahnges. The forge models commit has it's own branch in the tt-forge-models repo [jazpur/transformer-torch-uplift](https://github.com/tenstorrent/tt-forge-models/tree/jazpur/transformers-torch-uplift). 

Do the usual rebase in tt-xla:
```
git fetch origin
git rebase origin/main
```
You'll probably hit a conflict on the tt-forge-models commit since we uplift that repo into tt-xla almost daily. To resolve the conflicts do"
```
cd third_party/tt_forge_models
git fetch origin
git rebase <latest tt-forge-models commit on tt-xla>
## Resolve any conflicts if there are any

# We need to push the changes to the tt-forge-models repo
git push --force

# Go back to tt-xla and finish resolving the conflict
cd ../../
git add third_party/tt_forge_models
git rebase --continue
```
There may also be conflicts in tt-xla (the other commit), resolve those like usual and push.

## Testing

Once rebased, I would suggest running the `model-test-passing` job first, and solve any issues that may happen. Make sure to compare to the latest nightly run and only work on new failures.

Then you can test `basic-test-nightly`, and then `model-test-full` + `model-test-xfail`. I try not to run them all at the same time to avoid hogging all the runners.

## Updating tt-forge-models

If any changes are needed to the `tt-forge-models` repo, I recommend adding them to the already existing commit from tt-xla.

- Make changes in third_party/tt_forge_models
    ```
    cd third_party/tt_forge_models
    ## Make all the changes needed
    git add -u
    git commit --amend
    git push --force
    ```
- Commit changes to tt-xla using interactive rebase
    ```
    # Commit updated branch in tt-xla
    cd ../../
    git rebase -i HEAD~2
    ```
- In the editor, change pick to edit for the tt-forge-models commit:
    ```
    edit 05dc04b4 changes needed in tt-forge-models for transformers/torch updates
    pick 01a3104e update transformers library to 4.56.1 and torch to 2.9.0.
    ```
- Commit and finish:
    ```
    git add third_party/tt_forge_models
    git commit --amend
    git rebase --continue
    git push --force
    ```

## tt-forge

Once we are ready to merge, we will also need to merge some changes in tt-forge. We have a [PR](https://github.com/tenstorrent/tt-forge/pull/732) open that should be good to use once ready. 

Before that averything gets merged we should run a [Performance Benchmark CI](https://github.com/tenstorrent/tt-xla/actions/workflows/manual-benchmark.yml) job with the tt-xla + tt-forge uplift branches to make sure everything is still good, and once we have it passing we should also check for performance regressions to avoid the same issue we had this time. Contact Ognjen for any help on this.
