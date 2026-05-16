import MDXComponents from '@theme-original/MDXComponents';
import StatusBadges, {Badge} from '@site/src/components/StatusBadges';
import PageMeta from '@site/src/components/PageMeta';
import Todo from '@site/src/components/Todo';
import TBox from '@site/src/components/TBox';
import BrandName from '@site/src/components/BrandName';
import Terminal from '@site/src/components/Terminal';

export default {
  ...MDXComponents,
  StatusBadges,
  Badge,
  PageMeta,
  Todo,
  TBox,
  BrandName,
  ENABOL: BrandName,
  enabol: BrandName,
  Terminal,
};
