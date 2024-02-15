
const CONSTANT_VALUES = require('./constant');
 /** 
  * Invoked from staleCSAT.js and CSAT.yaml file to 
  * post survey link in closed issue.
  * @param {!object} Object -  gitHub and context contains the information about the
  *  current and context and github APIs using their built-in library functions.
 */
module.exports = async ({ github, context }) => {
    const issue = context.payload.issue.html_url;
    let base_url = CONSTANT_VALUES.MODULE.CSAT.BASE_URL;
     //Loop over all ths label present in issue and check if specific label is present for survey link.
    for (const label of context.payload.issue.labels) {
            if (CONSTANT_VALUES.MODULE.CSAT.CSAT_LABELS.includes(label.name)) {
                console.log(`label-${label.name}, posting CSAT survey for issue =${issue}`);
                const yesCsat = `<a href="${base_url + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
                    CONSTANT_VALUES.MODULE.CSAT.YES +
                    CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + issue}"> ${CONSTANT_VALUES.MODULE.CSAT.YES}</a>`;

                const noCsat = `<a href="${base_url + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
                    CONSTANT_VALUES.MODULE.CSAT.NO +
                    CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + issue}"> ${CONSTANT_VALUES.MODULE.CSAT.NO}</a>`;
                const comment = CONSTANT_VALUES.MODULE.CSAT.MSG + '\n' + yesCsat + '\n' +
                    noCsat + '\n';
                let issueNumber = context.issue.number ?? context.payload.issue.number;
                await github.rest.issues.createComment({
                    issue_number: issueNumber,
                    owner: context.repo.owner,
                    repo: context.repo.repo,
                    body: comment
                });
            }
        }
    };